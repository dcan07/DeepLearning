#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "hw2openmpkernel.h"

void kernel(unsigned int rows,unsigned int cols,float *data,float *w,float *results, unsigned int jobs){
	float dp=0;
	//get the tid
	int tid = omp_get_thread_num();
	if((tid+1)*jobs > rows) stop=rows;
        else stop = (tid+1)*jobs;
	int j;
	int i;
	int index;
	//Loop through jobs
	for (j = tid*jobs; j < stop; j++) {
		//calculate dp
		dp=0;
		index=cols * j;
        	for ( i = 0 ; i < cols ; i++ ) {
			dp =( dp+( w[i]*data[index]));
			index=index+1;
        	}
		//return dp
		results[j] = dp;

	}
	
}
