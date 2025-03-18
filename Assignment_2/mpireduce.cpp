// Reduction operations (sum, min, max, product, etc.) are useful for:

//     Summing an array distributed across multiple processes.
//     Finding the maximum/minimum element.
//     Computing the product of all elements.

// Implementation ->Parallel Sum of an Array
#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process has its own local value
    int local_value = rank + 1;  // Example: Process 0 has 1, Process 1 has 2, etc.

    int global_sum = 0;
    
    // Perform parallel reduction (sum operation)
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Root process prints the final result
    if (rank == 0) {
        cout << "Sum of all ranks: " << global_sum << endl;
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
