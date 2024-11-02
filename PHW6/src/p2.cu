#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 16
#define TILE_SIZE (N / nStreams) 

// Kernel to perform matrix multiplication on a tile
__global__ void matMulKernel(int *A, int *B, int *C, int A_offset, int C_offset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + A_offset;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int Cvalue = 0;
    for (int i = 0; i < N; ++i) {
        Cvalue += A[row * N + i] * B[i * N + col];
    }
    C[(row - A_offset) * N + col + C_offset] = Cvalue;
}

// Initialize matrix A and B with values as per the assignment
void init_matrices(int *A, int *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i;
            B[i * N + j] = j;
        }
    }
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);

    // Allocate host memory
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize matrices
    init_matrices(A, B);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy B to device synchronously as it is needed in full for each computation
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Create events and streams
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Start recording
    cudaEventRecord(startEvent);

    // First loop: copy slices of A to the device across all streams
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * N * TILE_SIZE;
        cudaMemcpyAsync(&d_A[offset], &A[offset], N * N * sizeof(int) / nStreams, cudaMemcpyHostToDevice, streams[i]);
    }

    // Second loop: kernel executions across all streams
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * TILE_SIZE;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((64 / nStreams), 64);
        matMulKernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A, d_B, d_C, offset, offset);
    }

    // Third loop: copy the result tiles from device to host across all streams
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * N * N / nStreams;
        cudaMemcpyAsync(&C[offset], &d_C[offset], N * N * sizeof(int) / nStreams, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Ensure all streams have finished their work before stopping the timer
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Stop recording
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("Time for computation and memory operations: %f ms\n", milliseconds);

    // Print the value of C[451][451]
    printf("Value of C[451][451]: %d\n", C[451 * N + 451]);
    printf("Value of C[0][0]: %d\n", C[0]);

    // Cleanup
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
