#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "hw2openmp.h"

int main(int argc ,char* argv[]) {
	//files for data and a vector
	FILE *path_data;
	FILE *path_w;
	
	//size of data
	size_t size;
	
	//other arguments
	unsigned int rows=atoi(argv[1]);
	unsigned int cols=atoi(argv[2]);
	int nprocs = atoi(argv[5]);
	
	//printf("rows=%d cols=%d nprocs =%d\n",rows,cols, numberprocs);


	//The dot product is returned here
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	
	//variables used
	unsigned int jobs; 
	unsigned long i;

	//size of data
	size = (size_t)((size_t)rows * (size_t)cols);
	//memory allocation to data and w
	float* data=(float*) malloc(sizeof(float)*size); 
	float* w=(float*) malloc(sizeof(float)*cols); 

	//read data
	fflush(stdout);
	path_data = fopen(argv[3], "r");
	if (path_data == NULL) {
    		printf("Cannot Open the File1");
		return 0;
	}
	for(i = 0; i <(rows*cols); i++){
		if (!fscanf(path_data, "%f", &data[i])) {break;}
	}
	//for(i = 0; i <(rows*cols); i++){printf("%f\n",data[i]);}
	fclose(path_data);
	fflush(stdout);

	//read a vector
	path_w = fopen(argv[4], "r");
	if (path_w == NULL) {
    		printf("Cannot Open the File2");
		return 0;
	}
	for(int b = 0; b <cols;b++){
	    if (!fscanf(path_w, "%f", &w[b])) {break;}
	}
	fclose(path_w);
	fflush(stdout);


	jobs = (unsigned int) ((cols+nprocs-1)/nprocs);

	//kernel function
	#pragma omp parallel num_threads(nprocs)
	kernel(rows,cols,data,w,host_results,jobs);


	//print the dp		
	for(int k = 0; k < rows; k++) {
		printf("%f", host_results[k]);
		printf("\n");
	}
	


	return 0;

}
