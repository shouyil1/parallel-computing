#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int size, rank;
    int msg;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        // Process 0 initializes Msg = 451 and prints the value of Msg
        msg = 451;
        printf("Process 0: Initially Msg = %d\n", msg);

        // Process 0 sends the value of Msg to Process 1
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);

        // Process 0 receives the value of Msg from Process 3 and prints the value
        MPI_Recv(&msg, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0: Received Msg = %d. Done!\n", msg);
    } 
    else {
        // Process 1, 2 and 3 logic
        MPI_Recv(&msg, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg += 1; // Increase the value by 1
        printf("Process %d: Msg = %d\n", rank, msg);
        
        // Send the current value of Msg to the next process
        MPI_Send(&msg, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
