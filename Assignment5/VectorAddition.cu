#include<iostream>
#include<chrono>
#include<cstdlib>

using namespace std;
using namespace std::chrono;

__global__ void addition(int *Darr, int *Dbrr, int *Dres, int n)
{
    int idx = (blockIdx.x * blockDim.x)+ threadIdx.x;
    if(idx < n)
    {
        Dres[idx]= Darr[idx]+Dbrr[idx];
    }
    __syncthreads();
}
void Display(int *a, int N)
{
    for(int i=0; i<N; i++)
        cout<<a[i]<<"\t";
    cout<<endl;
}
int main()
{
    int N;
    cout<<"Enter the size of vectors : ";
    cin>>N;

    int arr[N];
    int brr[N];
    int res[N];

    for(int i=0; i<N; i++)
    {
        arr[i] = rand() %10;
        brr[i] = rand() % 10;
    }

    int *darr;
    int *dbrr;
    int *dres;
    int size = N * sizeof(int);
    cudaMalloc((void **)&darr, size);
    cudaMalloc((void **)&dbrr, size);
    cudaMalloc((void **)&dres, size);

    cudaMemcpy(darr, arr, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dbrr, brr, size, cudaMemcpyHostToDevice);

    dim3 block((N/10) + 1, 1, 1);
    dim3 thread(10, 1, 1);

    addition<<<block, thread>>>(darr, dbrr, dres, N);
    cudaDeviceSynchronize();

    cudaMemcpy(res, dres, size, cudaMemcpyDeviceToHost);

    cout<<"First Vector : "<<endl;
    Display(arr, N);
    cout<<"Second Vector : "<<endl;
    Display(brr, N);
    cout<<"Result vector : "<<endl;
    Display(res, N);

    cudaFree(darr);
    cudaFree(dbrr);
    cudaFree(dres);
    return 0;
}