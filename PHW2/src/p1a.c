#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>


struct thread_data {
    int thread_id;
    double **A, **B, **C;
    int start_row, end_row, start_col, end_col, n;
};

void* matrix_multiply_thread(void* threadarg) {
    struct thread_data* data = (struct thread_data*) threadarg;

    for(int i = data->start_row; i < data->end_row; i++) {
        for(int j = data->start_col; j < data->end_col; j++) {
            for(int k = 0; k < data->n; k++) {
                data->C[i][j] += data->A[i][k] * data->B[k][j];
            }
        }
    }

    pthread_exit(NULL);
}


int main(int argc, char** argv){
		int i, j, k;
		struct timespec start, stop; 
		double time;
		int n = 4096; // matrix size is n*n
		int p = atoi(argv[1]); // size of thread
		
		if(n % p != 0) {
			printf("Block size must be a divisor of matrix size\n");
			return 1;
		}

		double **A = (double**) malloc (sizeof(double*)*n);
		double **B = (double**) malloc (sizeof(double*)*n);
		double **C = (double**) malloc (sizeof(double*)*n);
		for (i=0; i<n; i++) {
			A[i] = (double*) malloc(sizeof(double)*n);
			B[i] = (double*) malloc(sizeof(double)*n);
			C[i] = (double*) malloc(sizeof(double)*n);
		}
		
		for (i=0; i<n; i++){
			for(j=0; j< n; j++){
				A[i][j]=i;
				B[i][j]=i+j;
				C[i][j]=0;			
			}
		}
				
		if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
		
		// Your code goes here //
		// Matrix C = Matrix A * Matrix B //	
		//*******************************//
		pthread_t  threads[p*p];
		int block_size = n / p;
		struct thread_data thread_data_array[p*p];

		for(int row = 0; row < p; row++) {
			for(int col = 0; col < p; col++) {
				int tid = row * p + col;
				thread_data_array[tid].thread_id = tid;
				thread_data_array[tid].A = A;
				thread_data_array[tid].B = B;
				thread_data_array[tid].C = C;
				thread_data_array[tid].start_row = row * block_size;
				thread_data_array[tid].end_row = (row+1) * block_size;
				thread_data_array[tid].start_col = col * block_size;
				thread_data_array[tid].end_col = (col+1) * block_size;
				thread_data_array[tid].n = n;

				int rc = pthread_create(&threads[tid], NULL, matrix_multiply_thread, &thread_data_array[tid]);
				if (rc) {
					printf("ERROR; return code from pthread_create() is %d\n", rc);
					exit(-1);
				}
			}
		}

		// Join all threads back into the main thread
		for(int tid = 0; tid < p*p; tid++) {
			pthread_join(threads[tid], NULL);
		}
		
		//*******************************//
		
		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		
		printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", (unsigned long)2*n*n*n, time, 1/time/1e6*2*n*n*n);		
		printf("C[100][100]=%f\n", C[100][100]);
		
		// release memory
		for (i=0; i<n; i++) {
			free(A[i]);
			free(B[i]);
			free(C[i]);
		}
		free(A);
		free(B);
		free(C);
		return 0;
}
