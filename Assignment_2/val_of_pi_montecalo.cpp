// Randomly generating points in a square of size 2R × 2R.
// Counting how many fall inside the inscribed circle of radius R.
// Using the ratio:
// π≈4×Points Inside Circle / Total Points


// MPI Basic Functions Used here

// MPI_Init	
// MPI_Comm_rank	
// MPI_Comm_size	
// MPI_Reduce	
// MPI_Finalize	

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    int rank, size;
    long long local_hits = 0, global_hits = 0;
    long long total_points = 1000000;  // Total number of points
    long long local_points;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Seed the random number generator differently for each process
    srand(time(NULL) + rank * 100);

    // Divide work among processes
    local_points = total_points / size;

    // Monte Carlo Simulation
    for (long long i = 0; i < local_points; i++) {
        double x = (double)rand() / RAND_MAX;  // Random x in [0,1]
        double y = (double)rand() / RAND_MAX;  // Random y in [0,1]

        if (x * x + y * y <= 1.0) {
            local_hits++;
        }
    }

    // Reduce results to get the total number of points inside the circle
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Compute final estimation of Pi in root process
    if (rank == 0) {
        double pi_estimate = 4.0 * ((double)global_hits / total_points);
        cout << "Estimated Pi: " << pi_estimate << endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
