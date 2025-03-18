// Unlike MPI_Reduce, where only the root gets the result, 
// MPI_Allreduce distributes the final result to all processes.

#include <mpi.h>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank + 1;
    int global_sum = 0;

    // All processes get the final sum
    MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    cout << "Process " << rank << " received global sum: " << global_sum << endl;

    MPI_Finalize();
    return 0;
}
