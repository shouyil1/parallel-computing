#include <stdio.h>
#include <cublas.h>
#include <cuda_runtime.h>

#define N 1024 // Matrix size is 1024x1024
#define BLOCK_SIZE 32 // Block size 32x32 threads



__global__ void matrixMulShared(int *a, int *b, int *c) {
    int row = threadIdx.y;  // row in the block
	int col = threadIdx.x;  // col in the block
	int my_x = blockIdx.y * blockDim.y + threadIdx.y;  // row in the grid
	int my_y = blockIdx.x * blockDim.x + threadIdx.x;  // col in the grid

	int i,j;
	int local_c=0;
	__shared__ int A_s[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int B_s[BLOCK_SIZE][BLOCK_SIZE];

	for (i=0; i<N/BLOCK_SIZE; i++) {
		A_s[row][col] = a[my_x * N + (i * blockDim.y + col)];
		B_s[row][col] = b[(i * blockDim.x + row)*N + my_y];
		__syncthreads();
		for (j=0; j<BLOCK_SIZE; j++) {
			local_c += A_s[row][j] * B_s[j][col];
		}
		__syncthreads();
	}
	c[my_x*N + my_y] = local_c;
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);

    // Allocate host memory
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N*N; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

	struct timespec start, stop;
	double time;

    // Transfer data to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);

	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    // Execute the kernel
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("time is %f ns\n", time*1e9);

    // Print the result at C[451][451]
    printf("C[451][451] = %d\n", C[451 * N + 451]);

    // Cleanup
    free(A); 
	free(B); 
	free(C);
    cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_C);

    return 0;
}
