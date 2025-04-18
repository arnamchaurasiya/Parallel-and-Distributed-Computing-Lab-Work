/*
 Implement merge sort to sort the element of an array of the size n=1000.   
a. To implement parallelization use pipelining. 
b. Now implement the parallel merge sort using CUDA. 
c. Compare the performance of both (a) and (b) methods. 
*/

#include<iostream>
#include<cstdlib>
#include<chrono>
using namespace std;
using namespace std::chrono;

__device__ void conquer(int *arr, int start, int mid, int end)
{
    int *temp= new int[end-start+1];
    int i=start;
    int j= mid+1;
    int x=0;
    while(i<=mid && j<=end)
    {
        if(arr[i] <= arr[j])
            temp[x++]= arr[i++];
        else
            temp[x++]= arr[j++];
    }
    while(i<=mid)
        temp[x++]= arr[i++];
    while(j<=end)
        temp[x++]= arr[j++];
    for(int k= start,x=0; k<=end;)
        arr[k++]= temp[x++];
}

__device__ void divide(int *arr, int start, int end)
{
    if(start >= end)
        return;
    int mid= start + (end-start)/2;
    divide(arr, start, mid);
    divide(arr, mid+1, end);
    conquer(arr, start , mid, end);
}
__global__ void mergeSort(int *Darr, int n)
{
    int id= threadIdx.x;
    if(id == 0)
        divide(Darr, 0, n/2 -1);
    else    
        divide(Darr, n/2, n-1);
    __syncthreads();

    if(id == 0)
        conquer(Darr, 0, n/2-1, n-1);
}
int main()
{
    int N;
    cout<<"Enter the size of array : ";
    cin>>N;
    int *arr = new int[N];
    int *res = new int[N];
    for(int i=0; i<N; i++)
    {
        arr[i] = rand() % 10;
    }

    int *darr;
    int size = N * sizeof(int);
    cudaMalloc((void **)&darr, size);
    cudaMemcpy(darr, arr, size, cudaMemcpyHostToDevice);

    dim3 block(1, 1, 1);
    dim3 thread(2, 1, 1);

    // cout<<"Original Array : "<<endl;
    // for(int i=0; i<N; i++)
    //     cout<<arr[i]<<"\t";
    // cout<<endl;

    auto start = high_resolution_clock::now();
    mergeSort<<<block, thread>>>(darr, N);
    cudaDeviceSynchronize();
    auto end= high_resolution_clock::now();
    cudaMemcpy(res, darr, size, cudaMemcpyDeviceToHost);

    // cout<<"Sorted Array : "<<endl;
    // for(int i=0; i<N; i++)
    //     cout<<res[i]<<"\t";
    // cout<<endl;

    auto duration= duration_cast<nanoseconds>(end-start);
    cout<<"Time : "<<duration.count()<<endl;

    cudaFree(darr);
    return 0;
}