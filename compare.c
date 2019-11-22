#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

int main(int argc, char** argv) {
	FILE* file1 = fopen(argv[1], "rb");
	FILE* file2 = fopen(argv[2], "rb");
	//file1.len = file2.len
	uint32_t len1;
	uint32_t len2;
	fread(&len1, sizeof(uint32_t), 1, file1);
	fread(&len2, sizeof(uint32_t), 1, file2);
	if (len1 != len2) {
		printf("length not equal.\n");
		exit(0);
	}
	//find error value
	float temp1, temp2;
	temp1 = FLT_MIN;
	while (!feof(file1)) {
		fread(&temp2, sizeof(float), 1, file1);
		if (abs(temp2) > temp1)
			temp1 = abs(temp2);
	}
	float abstol = 1e-4 * temp1;
	float error = (abstol > 0.002 ? abstol : 0.002);
	printf("error tolerance:%f\n", error);
	uint32_t i = 1;
	fseek(file1, sizeof(uint32_t), SEEK_SET);
	while (!feof(file1)) {
		fread(&temp1, sizeof(float), 1, file1);
		fread(&temp2, sizeof(float), 1, file2);
		if (abs(temp1 - temp2) > error) {
			printf("%f,%f,not equal at %d.\n", temp1, temp2, i);
			break;
		}
		i++;
	}
}
