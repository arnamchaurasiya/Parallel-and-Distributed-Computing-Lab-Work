// MPI Prime Number Tester:

//Master Process (Rank 0)
//   Sends a number to each slave when requested.
//   Receives the results back from the slaves.
// Slave Processes
//Request a number from the master.
//   Test if it's prime.
//   Send back the number if prime, or its negative if not.

#include <mpi.h>
#include <iostream>

using namespace std;

// Function to check if a number is prime
bool is_prime(int num) {
    if (num <= 1) return false;
    if (num <= 3) return true;
    if (num % 2 == 0 || num % 3 == 0) return false;
    for (int i = 5; i * i <= num; i += 6) {
        if (num % i == 0 || num % (i + 2) == 0) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Master Process
        int numbers[] = {11, 15, 23, 40, 57, 97, 100}; // Numbers to test
        int numCount = sizeof(numbers) / sizeof(numbers[0]);

        for (int i = 1; i < size; i++) {
            int numIndex = i - 1;
            if (numIndex < numCount) {
                MPI_Send(&numbers[numIndex], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        // Receiving results from slaves
        for (int i = 1; i < size; i++) {
            int result;
            MPI_Recv(&result, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            cout << "Received from Process " << i << ": " << result << endl;
        }

    } else {
        // Slave Process
        int num;
        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        int result = is_prime(num) ? num : -num;
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
