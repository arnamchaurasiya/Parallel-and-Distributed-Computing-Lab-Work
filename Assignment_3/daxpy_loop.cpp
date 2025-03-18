#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define N (1 << 16)  // 2^16 elements
#define A 2.5        // Scalar multiplier

using namespace std;

void sequentialDAXPY(vector<double>& X, const vector<double>& Y, double a) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char** argv) {
    int rank, num_procs;
    double sequential_time = 0.0;  // Declare sequential_time outside

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_size = N / num_procs;
    vector<double> X(local_size), Y(local_size);

    if (rank == 0) {
        vector<double> full_X(N, 1.0), full_Y(N, 2.0);

        // Measure sequential execution time
        double seq_start = MPI_Wtime();
        sequentialDAXPY(full_X, full_Y, A);
        double seq_end = MPI_Wtime();
        sequential_time = seq_end - seq_start;  // Assign value

        cout << "Sequential Execution Time: " << sequential_time << " seconds\n";

        // Distribute data to processes
        MPI_Scatter(full_X.data(), local_size, MPI_DOUBLE, X.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(full_Y.data(), local_size, MPI_DOUBLE, Y.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, X.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(nullptr, local_size, MPI_DOUBLE, Y.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Measure parallel execution time
    double start_time = MPI_Wtime();

    for (int i = 0; i < local_size; i++) {
        X[i] = A * X[i] + Y[i];
    }

    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    // Gather results at rank 0
    if (rank == 0) {
        vector<double> result(N);
        MPI_Gather(X.data(), local_size, MPI_DOUBLE, result.data(), local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(X.data(), local_size, MPI_DOUBLE, nullptr, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Measure max execution time across all processes
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double speedup = sequential_time / max_time;  // Now it has a valid value
        cout << "Parallel Execution Time: " << max_time << " seconds\n";
        cout << "Speedup: " << speedup << "x\n";
    }

    MPI_Finalize();
    return 0;
}
