#include <stdio.h>
#include <cublas.h>

#define N 1024 // Matrix size is 1024x1024
#define BLOCK_SIZE 16 // Block size 16x16 threads

// CUDA kernel for matrix multiplication using global memory
__global__ void matrixMulGlobal(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int c_val = 0;
	for (int i = 0; i<N; i++) {
		c_val += a[row * N + i] * b[i * N + col];
	}
	c[row * N + col] = c_val;
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
	dim3 dimGrid(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
    // Execute the kernel
    matrixMulGlobal<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("time is %f ns\n", time*1e9);

    // Print the result at C[451][451]
    printf("C[451][451] = %d\n", C[451*N+451]);

    // Cleanup
    free(A); 
	free(B); 
	free(C);
    cudaFree(d_A); 
	cudaFree(d_B); 
	cudaFree(d_C);

    return 0;
}
