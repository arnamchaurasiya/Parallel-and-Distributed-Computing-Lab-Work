/*
Write a program using CUDA, in which all the threads are performing different 
tasks. These task are as follows:  
a. Find the sum of first n integer numbers. (you can take n as 1024. Do not use direct 
formula but use iterative approach)  
b. Find the sum of first n integer numbers. (you can take n as 1024. You can use direct 
formula not the iterative approach) 
*/

#include<iostream>
using namespace std;
#define N 1024

__global__ void calculate(int *Darr, int *Dres, int n)
{
    int id= threadIdx.x;
    if(id == 0)
    {
        for(int i=0; i<n; i++)
            Dres[0]+= Darr[i];
    }
    else
    {
        Dres[1]= (n* (n+1))/2;
    }
    __syncthreads();
}
int main()
{
    int arr[N];
    int res[2];
    for(int i=0; i<N; i++)
    {
        arr[i]= i+1;
    }

    int *darr;
    int *dres;
    cudaMalloc((void **)&darr, (N * sizeof(int)));
    cudaMalloc((void **)&dres, (2 * sizeof(int)));

    cudaMemcpy(darr, arr, (N * sizeof(int)), cudaMemcpyHostToDevice );

    dim3 block(1, 1, 1);
    dim3 thread(2, 1, 1);
    calculate<<<block, thread>>>(darr, dres, N);
    cudaDeviceSynchronize();

    cudaMemcpy(res, dres, (2 * sizeof(int)), cudaMemcpyDeviceToHost);

    cout<<"Result from thread 1 : "<<res[0]<<endl;
    cout<<"Result from thread 2 : "<<res[1]<<endl;

    cudaFree(darr);
    cudaFree(dres);
    return 0;
}