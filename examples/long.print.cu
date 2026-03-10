#include <cuda_runtime.h>
#include <stdio.h>

const int EXTRA = 5000000;

[[clang::annotate("polygeist_parallel_mgpu")]]
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float scale = 1.0f / (float)EXTRA;
        float s = 0.0f;
        for (int k = 0; k < EXTRA; k++)
            s += (A[i] + B[i]) * scale;
        C[i] = s;
    }
}

int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("Device count: %d\n", count);
    int n = 50 * 1024 * 1024;  // 50M elements
    float *d_A, *d_B, *d_C;
    cudaMallocManaged(&d_A, n * sizeof(float));
    cudaMallocManaged(&d_B, n * sizeof(float));
    cudaMallocManaged(&d_C, n * sizeof(float));

    float *h_A = (float *)malloc(n * sizeof(float));
    float *h_B = (float *)malloc(n * sizeof(float));
    float *h_C = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    printf("Launching vectorAdd kernel (%d elements, grid=%d, block=%d)...\n", n, grid, block);
    vectorAdd<<<grid, block>>>(d_A, d_B, d_C, n);
    for (int d = 0; d < count; d++) {
        cudaSetDevice(d);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
    printf("Kernel completed.\n");

    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("C[0] = %.2f (expected 0)\n", h_C[0]);
    printf("C[1] = %.2f (expected 3)\n", h_C[1]);
    int last = n - 1;
    float expectedLast = (float)(3 * last);
    printf("C[%d] = %.2f (expected %.0f)\n", last, h_C[last], expectedLast);
    const float relTol = 0.1f;  // 10%
    int ok = (h_C[0] == 0.0f &&
              h_C[1] >= 3.0f * (1.0f - relTol) && h_C[1] <= 3.0f * (1.0f + relTol) &&
              h_C[last] >= expectedLast * (1.0f - relTol) && h_C[last] <= expectedLast * (1.0f + relTol));
    printf("%s\n", ok ? "SUCCESS: vectorAdd results correct." : "FAIL: mismatch.");

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return ok ? 0 : 1;
}