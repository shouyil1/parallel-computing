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
    for (int k = 0; k < N; ++k) {
        Cvalue += A[row * N + k] * B[k * N + col];
    }
    C[(row-A_offset) * N + col + C_offset] = Cvalue;
}


// Main program
int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(int);
    // int nStreams = 4;
    // int TILE_SIZE = (N / nStreams);

    // Allocate host memory
    cudaMallocHost((void**)&A, size);
    cudaMallocHost((void**)&B, size);
    cudaMallocHost((void**)&C, size);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i;
            B[i * N + j] = j;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy B to device synchronously as it is needed in full for each computation
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Start recording
    cudaEventRecord(startEvent, 0);

    // Loop over streams
    for (int i = 0; i < nStreams; ++i) {
        // Calculate offsets
        int A_offset = i * TILE_SIZE;
        int C_offset = i * TILE_SIZE * N;

        // Asynchronously copy a tile of A to the device
        cudaMemcpyAsync(&d_A[A_offset * N], &A[A_offset * N], N * N * sizeof(int) / nStreams, cudaMemcpyHostToDevice, streams[i]);

        // Configure grid and block dimensions
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((64 / nStreams), 64);

        // Launch the kernel on a stream
        matMulKernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A, d_B, d_C, A_offset, C_offset);

        // Check for any errors in kernel launch
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        }

        // Asynchronously copy a tile of C back to the host
        cudaMemcpyAsync(&C[C_offset], &d_C[C_offset], N * N * sizeof(int) / nStreams, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Stop recording
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("Time for computation and memory operations: %f ms\n", milliseconds);

    // Print the value of C[451][451]
    printf("Value of C[451][451]: %d\n", C[451 * N + 451]);

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(dummyEvent);
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
