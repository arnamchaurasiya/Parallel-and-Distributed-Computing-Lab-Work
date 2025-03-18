#include <mpi.h>
#include <iostream>
#include <cmath>

using namespace std;

int main(int argc, char* argv[]) {
    int rank, size;
    long num_steps = 100000;  // Number of steps for approximation
    double step, sum = 0.0, pi, local_sum = 0.0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Broadcast num_steps to all processes
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    step = 1.0 / (double)num_steps;

    // Each process computes a portion of the sum
    for (long i = rank; i < num_steps; i += size) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Reduce all partial sums to compute the final pi value
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = step * sum;
        cout << "Computed value of Pi: " << pi << endl;
        cout << "Error: " << fabs(M_PI - pi) << endl;
    }

    MPI_Finalize();
    return 0;
}