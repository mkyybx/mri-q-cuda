
#include <stdio.h>
#include <pthread.h>

#define BLOCK_SIZE 64
#define REDUCTION_BLOCK_SIZE 1024
#define PIx2 6.2831853071795864769252867665590058f
#include <sys/time.h>

struct kValues {
	float Kx;
	float Ky;
	float Kz;
	float PhiMag;
};


//size needed: numK * 1
__global__ void ComputePhiMagGPU(struct kValues* kValsD, float* phiRD, float* phiID) {
	int indexK = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	//Shared memory is not needed since this is a coalesced access.
	//kVals.KxKyKz should be initialized in the host since it's pure memory operation. CUDA is not used for doing parrallel data memory operation.
	kValsD[indexK].PhiMag = phiRD[indexK] * phiRD[indexK] + phiID[indexK] * phiID[indexK];
}

__global__ void ImprovedReductionKernel(float* globalData, int interval, int dataSize) {
	int loc = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float data[REDUCTION_BLOCK_SIZE];
	if (loc * interval < dataSize) {
		//load to shared mem
		data[threadIdx.x] = globalData[loc];
		int stride = REDUCTION_BLOCK_SIZE / 2;
		do {
			__syncthreads();
			/*if (threadIdx.x == 0)
				printf("datasize=%d\n", dataSize);*/
			if (threadIdx.x < stride && threadIdx.x + stride < dataSize) {
				data[threadIdx.x] += data[threadIdx.x + stride];
				/*printf("%f,", data[threadIdx.x]);*/
			}
			stride >>= 1;
		} while (stride >= 1);
		if (threadIdx.x == 0) {
			globalData[loc] = data[0];
		}
	}
}

//size needed: numK * 1
__global__ void ComputeQGPU(float* globalqr, float* globalqi, struct kValues* globalkVals, float globalx, float globaly, float globalz) {
	//constant memory will limit the scalibility
	__shared__ float x, y, z;
	__shared__ struct kValues kVals[BLOCK_SIZE];
	__shared__ float Qracc[BLOCK_SIZE];
	__shared__ float Qiacc[BLOCK_SIZE];
	int indexK = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	//load shared mem
	kVals[threadIdx.x] = globalkVals[indexK];

	if (threadIdx.x == 0) {
		x = globalx;
		y = globaly;
		z = globalz;
	}
	__syncthreads();
	float expArg = PIx2 * (kVals[threadIdx.x].Kx * x + kVals[threadIdx.x].Ky * y + kVals[threadIdx.x].Kz * z);
	float cosArg, sinArg;
	sincosf(expArg, &sinArg, &cosArg);
	//the following should be zero for padding
	Qracc[threadIdx.x] = kVals[threadIdx.x].PhiMag * cosArg;
	Qiacc[threadIdx.x] = kVals[threadIdx.x].PhiMag * sinArg;

	//improved reduction
	int stride = BLOCK_SIZE / 2;
	do {
		__syncthreads();
		if (threadIdx.x < stride) {
			Qracc[threadIdx.x] += Qracc[threadIdx.x + stride];
			Qiacc[threadIdx.x] += Qiacc[threadIdx.x + stride];
		}
		stride >>= 1;
	} while (stride >= 1);
	if (threadIdx.x == 0) {
		*(globalqr + blockIdx.x) = Qracc[0];
		*(globalqi + blockIdx.x) = Qiacc[0];
	}
}

//cudaMalloc inside
void launchKernel(int numK, int numX, float* kxH, float* kyH, float* kzH, 
							float* xH, float* yH, float* zH, float* phiRH, float* phiIH, float* QrH, float* QiH,
							float** phiRD, float** phiID, struct kValues** kValsD) {
    struct timeval time0;
    struct timeval time1;
    struct timezone tz;
//    long kernelTime = 0;
//    long memoryTime = 0;
	//calculate dimension
	dim3 dim_grid, dim_block;
	dim_grid.x = numK / BLOCK_SIZE + (numK % BLOCK_SIZE == 0 ? 0 : 1);
	dim_grid.y = 1;
	dim_grid.z = 1;
	dim_block.x = BLOCK_SIZE;
	dim_block.y = 1;
	dim_block.z = 1;
	fflush(stdout);
	//prepare for calculating PhiMag
	cudaMalloc(kValsD, dim_grid.x * BLOCK_SIZE * sizeof(struct kValues));
	struct kValues* kVals = (struct kValues*)calloc(numK, sizeof(struct kValues));
	for (int k = 0; k < numK; k++) {
		kVals[k].Kx = kxH[k];
		kVals[k].Ky = kyH[k];
		kVals[k].Kz = kzH[k];
	}
//    gettimeofday(&time0, &tz);
	cudaMemset(*kValsD, 0, numK * sizeof(struct kValues));
	cudaMemcpy(*kValsD, kVals, numK * sizeof(struct kValues), cudaMemcpyHostToDevice);

	cudaMalloc(phiRD, dim_grid.x * BLOCK_SIZE * sizeof(struct kValues));
	cudaMemset(*phiRD, 0, numK * sizeof(float)); //0 * n = 0
	cudaMemcpy(*phiRD, phiRH, numK * sizeof(struct kValues), cudaMemcpyHostToDevice);

	cudaMalloc(phiID, dim_grid.x * BLOCK_SIZE * sizeof(struct kValues));
	cudaMemcpy(*phiID, phiIH, numK * sizeof(struct kValues), cudaMemcpyHostToDevice);
//    gettimeofday(&time1, &tz);
//    memoryTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;

	//calculate phiMag
//	gettimeofday(&time0, &tz);
	ComputePhiMagGPU<<<dim_grid, dim_block>>> (*kValsD, *phiRD, *phiID);
	cudaDeviceSynchronize();
//    gettimeofday(&time1, &tz);
//    kernelTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;

    //launch kernel
	//multithreading could be used, but it's not necessary. Even 32*32*32 input(numK=3072) would occupy all threads (2560 for RTX2070) simultaneously, which
	//use around 2s of CPU. Multithreading would help if there are small inputs, but why not just do it on CPU?
	//multithreading will decrease 32x32x32 performance by half
	for (int indexX = 0; indexX < numX; indexX++) {

		//allocate result space. per indexX
		float* globalqrD;
		float* globalqiD;

//        gettimeofday(&time0, &tz);
		cudaMalloc(&globalqrD, dim_grid.x * sizeof(float));
		cudaMalloc(&globalqiD, dim_grid.x * sizeof(float));
//        gettimeofday(&time1, &tz);
//        memoryTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;

//        gettimeofday(&time0, &tz);
		ComputeQGPU<<<dim_grid, dim_block>>>(globalqrD, globalqiD, *kValsD, xH[indexX], yH[indexX], zH[indexX]);
        cudaDeviceSynchronize();
//        gettimeofday(&time1, &tz);
//        kernelTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;

		//reduction
		int currentDataNum = dim_grid.x;
		int interval = 1;
		dim3 dim_grid_reduction, dim_block_reduction;
		while (currentDataNum != 1) {
			dim_grid_reduction.x = currentDataNum / REDUCTION_BLOCK_SIZE + (currentDataNum % REDUCTION_BLOCK_SIZE == 0 ? 0 : 1);
			dim_grid_reduction.y = 1;
			dim_grid_reduction.z = 1;
			dim_block_reduction.x = REDUCTION_BLOCK_SIZE;
			dim_block_reduction.y = 1;
			dim_block_reduction.z = 1;
//            gettimeofday(&time0, &tz);
			ImprovedReductionKernel<<<dim_grid_reduction, dim_block_reduction>>>(globalqrD, interval, currentDataNum);
			ImprovedReductionKernel<<<dim_grid_reduction, dim_block_reduction>>>(globalqiD, interval, currentDataNum);
			cudaDeviceSynchronize();
//            gettimeofday(&time1, &tz);
//            kernelTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;
			interval *= REDUCTION_BLOCK_SIZE;
			currentDataNum = currentDataNum / REDUCTION_BLOCK_SIZE + (currentDataNum % REDUCTION_BLOCK_SIZE == 0 ? 0 : 1);
		}

//        gettimeofday(&time0, &tz);
		cudaMemcpy(&(QrH[indexX]), globalqrD, sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&(QiH[indexX]), globalqiD, sizeof(float), cudaMemcpyDeviceToHost);
//        gettimeofday(&time1, &tz);
//        memoryTime += (time1.tv_sec - time0.tv_sec) * 1000000 + time1.tv_usec - time0.tv_usec;
	}

//    printf("kernel: %ld us\n", kernelTime);
//    printf("IO: %ld us\n", memoryTime);


}
