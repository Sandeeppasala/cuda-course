#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 512

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void vector_add_gpu(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float* __restrict__ C,
                   int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process pairs
    for (int idx = tid; idx < n / 4; idx += stride) {
        float4 x = reinterpret_cast<const float4*>(A)[idx];
        float4 y = reinterpret_cast<const float4*>(B)[idx];

        float4 z;
        z.x = __fadd_rn(x.x, y.x);
        z.y = __fadd_rn(x.y, y.y);
        z.z = __fadd_rn(x.z, y.z);
        z.w = __fadd_rn(x.w, y.w);

        reinterpret_cast<float4*>(C)[idx] = z;
    }

    // Handle tail elements (single thread only)
    int tail = n & 3;
    if (tail && tid == 0) {
        for (int i = n - tail; i < n; ++i) {
            C[i] = __fadd_rn(A[i], B[i]);
        }
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate pinned host memory
    cudaHostAlloc((void**)&h_a, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c_cpu, size, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c_gpu, size, cudaHostAllocDefault);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device (async)
    cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream); // Ensure data is on device before kernel

#if 0
    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
#else
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int num_blocks = numSMs * 16;   // or *16 or *32
#endif

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(s, stream);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_c, N);
        cudaEventRecord(e, stream);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        printf("Kernel time: %f ms\n", ms);
        gpu_total_time += ms / 1000.0;
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    cudaMemcpyAsync(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c_cpu);
    cudaFreeHost(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaStreamDestroy(stream);
    return 0;
}
