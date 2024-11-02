#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;

    // Initialize the MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int numberArray[64];
    int localData[16];  // Each process will have 1/4th of the array

    // Process 0 reads the array
    if (rank == 0) {
        FILE *numberFile = fopen("number.txt", "r");
        for (int i = 0; i < 64; i++) {
            fscanf(numberFile, "%d", &numberArray[i]);
        }
        fclose(numberFile);
    }

    // Process 0 scatters the array to every process
    MPI_Scatter(numberArray, 16, MPI_INT, localData, 16, MPI_INT, 0, MPI_COMM_WORLD);

    // each processor compute their localSum
    int localSum = 0;
    for (int i = 0; i < 16; i++) {
        localSum += localData[i];
    }

    // gather partial sums
    int gatherData[4];  
    MPI_Gather(&localSum, 1, MPI_INT, gatherData, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 computes the sum of all the partial sums and prints it out.
    if (rank == 0) {
        int globalSum = 0;
        for (int i = 0; i < 4; i++) {
            globalSum += gatherData[i];
        }
        printf("Process %d has Total sum: %d\n", rank, globalSum);
    }

    MPI_Finalize();

    return 0;
}
