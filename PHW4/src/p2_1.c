#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    
    // Initialize the MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // read inputs
    FILE *numberFile;
    numberFile = fopen("number.txt", "r");
    int numberArray[64];
    for (int i = 0; i < 64; i++) {
        fscanf(numberFile, "%d", &numberArray[i]);
    }
    fclose(numberFile);

    // each processor compute their localSum
    int localSum = 0;
    int start_idx = rank * 16;
    int end_idx = start_idx + 16;
    for (int i = start_idx; i < end_idx; i++) {
        localSum += numberArray[i];
    }

    // Process 1,2,3 send their partial sum to Process 0
    int globalSum;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Process 0 prints it out
    if (rank == 0) {
        printf("Process %d has Total sum: %d\n", rank, globalSum);
    }

    MPI_Finalize();

    return 0;
}
