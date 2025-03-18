// Steps in the Program

//     Generate two 70x70 matrices with random values.
//     Perform matrix multiplication sequentially and measure the time.
//     Perform matrix multiplication in parallel using MPI, dividing rows among processes.
//     Compare execution times of sequential vs. parallel execution.

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>

#define SIZE 70  // Matrix size

using namespace std;

void generateMatrix(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            matrix[i][j] = rand() % 10;  // Random values (0-9)
}

// Serial Matrix Multiplication
void sequentialMultiply(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// Parallel Matrix Multiplication using MPI
void parallelMultiply(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE], int rank, int size) {
    int rows_per_process = SIZE / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? SIZE : start_row + rows_per_process;

    for (int i = start_row; i < end_row; i++)
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];
    double start_time, run_time;

    // Root process generates matrices
    if (rank == 0) {
        srand(42);  // Seed for reproducibility
        generateMatrix(A);
        generateMatrix(B);
    }

    // Broadcast B to all processes
    MPI_Bcast(B, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Sequential Execution (Only on Rank 0)
    if (rank == 0) {
        start_time = omp_get_wtime();
        sequentialMultiply(A, B, C);
        run_time = omp_get_wtime() - start_time;
        cout << "Sequential Execution Time: " << run_time << " seconds" << endl;
    }

    // Start Parallel Execution
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = omp_get_wtime();
    MPI_Bcast(A, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process computes part of the result
    parallelMultiply(A, B, C, rank, size);

    // Gather results from all processes
    if (rank == 0) {
    MPI_Gather(MPI_IN_PLACE, (SIZE / size) * SIZE, MPI_DOUBLE,
               C, (SIZE / size) * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               } 
else {
    MPI_Gather(C[rank * (SIZE / size)], (SIZE / size) * SIZE, MPI_DOUBLE,
               C, (SIZE / size) * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               }


    // Compute Parallel Execution Time
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        run_time = omp_get_wtime() - start_time;
        cout << "Parallel Execution Time using " << size << " processes: " << run_time << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
