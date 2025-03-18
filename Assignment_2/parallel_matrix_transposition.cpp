
// Distribute the matrix row-wise across processes.
// Each process transposes its assigned rows into a local buffer.
// Use MPI_Gather or MPI_Allgather to collect the transposed parts.
#include <mpi.h>
#include <iostream>
#include <vector>

#define N 4  // Matrix size (N x N)

using namespace std;

// Function to print the matrix
void printMatrix(const vector<int>& mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << mat[i * N + j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char** argv) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    vector<int> matrix(N * N, 0);
    vector<int> transposed(N * N, 0);

    if (rank == 0) {
        cout << "Original Matrix:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = i * N + j + 1;
            }
        }
        printMatrix(matrix);
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(matrix.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process transposes a part of the matrix
    for (int i = rank; i < N; i += num_procs) {
        for (int j = 0; j < N; j++) {
            transposed[j * N + i] = matrix[i * N + j];
        }
    }

    // Gather the transposed parts from all processes
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : transposed.data(), transposed.data(),
               N * N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nTransposed Matrix:\n";
        printMatrix(transposed);
    }

    MPI_Finalize();
    return 0;
}
