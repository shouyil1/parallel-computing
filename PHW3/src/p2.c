#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define		size	   2*256*256


int partition(int *array, int start, int end) {
    int pivot = array[end];
    int i = start - 1;
    for(int j = start; j <= end - 1; j++) {
        if(array[j] < pivot) {
            i++;
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    int temp = array[i + 1];
    array[i + 1] = array[end];
    array[end] = temp;
    return i + 1;
}

int randomizedPartition(int *array, int start, int end) {
    int randomIndex = start + rand() % (end - start + 1); // random index between start and end
    // swap array[end] with array[randomIndex]
    int temp = array[end];
    array[end] = array[randomIndex];
    array[randomIndex] = temp;
    return partition(array, start, end);
}

void quickSort(int *array, int start, int end){
   // you quick sort function goes here  
    if(start < end) {
        int pIndex = partition(array, start, end);
        quickSort(array, start, pIndex - 1);
        quickSort(array, pIndex + 1, end);
    }
}

void parallelQuickSort(int *array, int start, int end) {
    if(start < end) {
        int pIndex = randomizedPartition(array, start, end);
		omp_set_num_threads(2);
        #pragma omp sections
        {
            #pragma omp section
            {
                quickSort(array, start, pIndex - 1);
            }
            #pragma omp section
            {
                quickSort(array, pIndex + 1, end);
            }
        }
    }
}


int main(void){
	int i, j, tmp;
	struct timespec start, stop; 
	double exe_time;
	srand(time(NULL)); 
	int * m = (int *) malloc (sizeof(int)*size);
	for(i=0; i<size; i++){
		//m[i]=size-i;
		m[i] = rand();
	}
	
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	////////**********Your code goes here***************//
	
	parallelQuickSort(m, 0, size-1);
			
	///////******************************////
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	exe_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	for(i=0;i<16;i++) printf("%d ", m[i]);		
	printf("\nExecution time = %f sec\n",  exe_time);	
}	