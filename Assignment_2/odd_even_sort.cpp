// Generate an array of random numbers (on rank 0).
// Distribute the array among processes.
// Perform Odd-Even Sorting in parallel:

//     Odd and even phases of sorting happen in iterations.
//     Each process exchanges and sorts elements with its neighbor.

// Gather the sorted subarrays back to rank 0.
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>

using namespace std;

void compareSplit(vector<int> &local_data, vector<int> &recv_data, bool keep_small) {
    vector<int> merged(local_data.size() * 2);
    merge(local_data.begin(), local_data.end(), recv_data.begin(), recv_data.end(), merged.begin());
    
    if (keep_small)
        copy(merged.begin(), merged.begin() + local_data.size(), local_data.begin());
    else
        copy(merged.end() - local_data.size(), merged.end(), local_data.begin());
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int n = 16;  // Total number of elements (should be divisible by number of processes)
    int local_n = n / size;
    vector<int> local_data(local_n);
    
    srand(rank + time(0));  // Seed for randomness
    for (int i = 0; i < local_n; i++)
        local_data[i] = rand() % 100;  // Random values between 0 and 99
    
    sort(local_data.begin(), local_data.end());  // Local sorting
    
    cout << "Process " << rank << " initial array: ";
    for (int num : local_data) cout << num << " ";
    cout << endl;
    
    for (int phase = 0; phase < size; phase++) {
        int partner = (phase % 2 == 0) ? (rank % 2 == 0 ? rank + 1 : rank - 1)
                                      : (rank % 2 == 0 ? rank - 1 : rank + 1);
        
        if (partner >= 0 && partner < size) {
            vector<int> recv_data(local_n);
            MPI_Sendrecv(local_data.data(), local_n, MPI_INT, partner, 0,
                         recv_data.data(), local_n, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            compareSplit(local_data, recv_data, rank < partner);
        }
    }
    
    vector<int> sorted_data;
    if (rank == 0) sorted_data.resize(n);
    MPI_Gather(local_data.data(), local_n, MPI_INT,
               sorted_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\nSorted Array: ";
        for (int num : sorted_data) cout << num << " ";
        cout << endl;
    }
    
    MPI_Finalize();
    return 0;
}