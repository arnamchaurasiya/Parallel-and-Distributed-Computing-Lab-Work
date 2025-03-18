// The prefix sum (scan) computes the cumulative sum of an array in parallel. MPI provides two functions for this:

// MPI_Scan – Each process gets a partial sum of all preceding values.

#include <mpi.h>
#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_value = rank + 1;  // Each process has a value
    int prefix_sum = 0;          // Store partial prefix sum

    // Perform parallel prefix sum using MPI_Scan
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Print results from each process
    cout << "Process " << rank << " - Local Value: " << local_value
         << ", Prefix Sum: " << prefix_sum << endl;

    MPI_Finalize();
    return 0;
}
