#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include "solution.h"

int main(int argc ,char* argv[]) {
	//files for data and a vector
	FILE *fp;
	FILE *fpw;
	
	//size of data
	size_t size;
	
	//other arguments
	unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	int CUDA_DEVICE = atoi(argv[5]);
	int THREADS = atoi(argv[6]);
	
	//printf("rows=%d cols=%d CUDA_DEVICE=%d\n",rows,cols,CUDA_DEVICE);
	//set device
	cudaError err = cudaSetDevice(CUDA_DEVICE);
	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	int BLOCKS;
	//The dot product is returned here
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	
	//variables used
	unsigned int jobs; 
	unsigned long i;
	unsigned long j;
	unsigned long ind;

	/*Kernel variable declaration */
	float *dev_data;
	float *results;
	float *dev_w;

	//size of data
	size = (size_t)((size_t)rows * (size_t)cols);
	//memory allocation to data and w
	float* data=(float*) malloc(sizeof(float)*size); 
	float* w=(float*) malloc(sizeof(float)*cols); 

	//read data as transpose
	//the indices for transpose is calculated in the loop
	fflush(stdout);
	fp = fopen(argv[3], "r");
	if (fp == NULL) {
    		printf("Cannot Open the File1");
		return 0;
	}
	for(i = 0; i <rows; i++){
		for(j = 0; j < cols; j++){
			ind=(j*rows)+i;
			if (!fscanf(fp, "%f", &data[ind])) {break;}
		}
	}
	//for(i = 0; i <(rows*cols); i++){printf("%f\n",data[i]);}
	fclose(fp);
	fflush(stdout);

	//read a vector
	fpw = fopen(argv[4], "r");
	if (fpw == NULL) {
    		printf("Cannot Open the File2");
		return 0;
	}
	for(int b = 0; b <cols;b++){
	    if (!fscanf(fpw, "%f", &w[b])) {break;}
	}
	fclose(fpw);
	fflush(stdout);

	//CUDA part of the code
	err = cudaMalloc((float**)&dev_data, size *  sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }

	err = cudaMalloc((float**) &results, rows * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
	
	err = cudaMalloc((float**) &dev_w, cols * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }

	err = cudaMemcpy(dev_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
	
	err = cudaMemcpy(dev_w, w, cols * sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }

	jobs = rows;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

	//kernel function
	kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_data,dev_w,results);
		
	// copy back
	cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);

	//print the dp		
	for(int k = 0; k < jobs; k++) {
		printf("%f", host_results[k]);
		printf("\n");
	}
	
	cudaFree( dev_data );
	cudaFree( results );
	cudaFree( dev_w );

	return 0;

}
