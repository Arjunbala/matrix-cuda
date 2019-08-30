/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 16
#endif

#ifndef BATCH_SIZE
    #define BATCH_SIZE 1
#endif

#ifndef NUM_ITERATIONS
    #define NUM_ITERATIONS 1024
#endif

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

/*
*********************************************************************
function name: main

description: test and compare

parameters: 
            none

return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    int m1, n1, k1, m2, n2, k2;

    /* Fixed seed for illustration */
    srand(3333);
    
    m1=BATCH_SIZE;
    n1=65536;
    k1=1024;

    m2=BATCH_SIZE;
    n2=1024;
    k2=10;

    // allocate memory in host RAM
    int *h_a, *h_b, *h_c, *h_d, *h_e;
    cudaMallocHost((void **) &h_a, sizeof(int)*m1*n1);
    cudaMallocHost((void **) &h_b, sizeof(int)*n1*k1);
    cudaMallocHost((void **) &h_c, sizeof(int)*m1*k1);
    cudaMallocHost((void **) &h_d, sizeof(int)*n2*k2);
    cudaMallocHost((void **) &h_e, sizeof(int)*m2*k2);

    // random initialize matrix B
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < k1; ++j) {
            h_b[i * k1 + j] = rand() % 1024;
        }
    }

    // random initialize matrix D
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < k2; ++j) {
            h_b[i * k2 + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory space on the device 
    int *d_a, *d_b, *d_c, *d_d, *d_e;
    cudaMalloc((void **) &d_a, sizeof(int)*m1*n1);
    cudaMalloc((void **) &d_b, sizeof(int)*n1*k1);
    cudaMalloc((void **) &d_c, sizeof(int)*m1*k1);
    cudaMalloc((void **) &d_d, sizeof(int)*n2*k2);
    cudaMalloc((void **) &d_e, sizeof(int)*m2*k2);

    // copy matrix B,D from host to device memory - these are weight matrices
    cudaMemcpy(d_b, h_b, sizeof(int)*n1*k1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, sizeof(int)*n2*k2, cudaMemcpyHostToDevice);

    int numExamples = 0;
    double total_time_ms = 0.0;
    for(int i=0;i<NUM_ITERATIONS;i++) {
        // random initialize matrix A - this is the input matrix
        for (int i = 0; i < m1; ++i) {
            for (int j = 0; j < n1; ++j) {
                h_a[i * n1 + j] = rand() % 1024;
            }
        }
	cudaEventRecord(start, 0);
	// copy from host to device
        cudaMemcpy(d_a, h_a, sizeof(int)*m1*n1, cudaMemcpyHostToDevice);
	
	unsigned int grid_rows = (m1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

        // Launch kernel for multiplication 1
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m1, n1, k1);
        cudaDeviceSynchronize();

        // Launch kernel for multiplication 2
        grid_rows = (m2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid_cols = (k2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid2(grid_cols, grid_rows);
        dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        gpu_matrix_mult<<<dimGrid2, dimBlock2>>>(d_c, d_d, d_e, m2, n2, k2);

        // Transefr results from device to host
        cudaMemcpy(h_e, d_e, sizeof(int)*m2*k2, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapse on GPU computing
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
	numExamples += BATCH_SIZE;
	total_time_ms += gpu_elapsed_time_ms;
    }
    printf("Avg. Latency: %g ms :: Avg. Throughput: %g examples/sec\n", total_time_ms/NUM_ITERATIONS, numExamples*1000.0/total_time_ms);

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_d);
    cudaFreeHost(h_e);
    return 0;
}
