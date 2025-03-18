// Distribute the two vectors among processes.
// Each process computes the dot product for its portion.
// Use MPI_Reduce() or MPI_Allreduce() to sum up the partial results.

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding random generator

using namespace std;

#define VECTOR_SIZE 100  // Size of vectors

int main(int argc, char** argv) {
    int rank, num_procs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_size = VECTOR_SIZE / num_procs; // Size of chunk per process
    vector<int> A(local_size);
    vector<int> B(local_size);
    vector<int> full_A, full_B;  // Only used in rank 0

    // Rank 0 initializes the full vectors
    if (rank == 0) {
        full_A.resize(VECTOR_SIZE);
        full_B.resize(VECTOR_SIZE);
        srand(time(0));
        for (int i = 0; i < VECTOR_SIZE; i++) {
            full_A[i] = rand() % 10;  // Random values 0-9
            full_B[i] = rand() % 10;
        }
    }

    // Scatter vectors to all processes
    MPI_Scatter(full_A.data(), local_size, MPI_INT, A.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(full_B.data(), local_size, MPI_INT, B.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local dot product
    int local_dot_product = 0;
    for (int i = 0; i < local_size; i++) {
        local_dot_product += A[i] * B[i];
    }

    // Reduce to compute final dot product
    int global_dot_product = 0;
    MPI_Reduce(&local_dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final result
    if (rank == 0) {
        cout << "Final Dot Product: " << global_dot_product << endl;
    }

    MPI_Finalize();
    return 0;
}
