#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include "solution.h"

__global__ void kernel(unsigned int rows,unsigned int cols,float *snpdata,float *w,float *results){
	float dp=0;
	//get the tid
	unsigned int tid =threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int j;
	unsigned int index;
	//calculate dp
        for ( j = 0 ; j < cols ; j++ ) {
		//calculate the current data index
		//consecutive threads access consecutive threads
	        index=(tid +( rows * j));
		dp =( dp+( w[j]*snpdata[index]));
        }
	//return dp
	results[tid] = dp;
}
