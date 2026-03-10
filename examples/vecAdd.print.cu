#include <cuda_runtime.h>
#include <stdio.h>

[[clang::annotate("polygeist_parallel_mgpu")]]
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("Device count: %d\n", count);
    int n = 1024;
    float *d_A, *d_B, *d_C;
    cudaMallocManaged(&d_A, n * sizeof(float));
    cudaMallocManaged(&d_B, n * sizeof(float));
    cudaMallocManaged(&d_C, n * sizeof(float));
    // cudaMalloc(&d_A, n * sizeof(float));
    // cudaMalloc(&d_B, n * sizeof(float));
    // cudaMalloc(&d_C, n * sizeof(float));

    float *h_A = (float *)malloc(n * sizeof(float));
    float *h_B = (float *)malloc(n * sizeof(float));
    float *h_C = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    printf("Launching vectorAdd kernel (%d elements)...\n", n);
    vectorAdd<<<32, 32>>>(d_A, d_B, d_C, n);
    for (int d = 0; d < count; d++) {
        cudaSetDevice(d);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
    printf("Kernel completed.\n");

    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0] = %.0f (expected 0)\n", h_C[0]);
    printf("C[1] = %.0f (expected 3)\n", h_C[1]);
    printf("C[1023] = %.0f (expected 3069)\n", h_C[1023]);
    int ok = (h_C[0] == 0.0f && h_C[1] == 3.0f && h_C[1023] == 3069.0f);
    printf("%s\n", ok ? "SUCCESS: vectorAdd results correct." : "FAIL: mismatch.");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ok ? 0 : 1;
}