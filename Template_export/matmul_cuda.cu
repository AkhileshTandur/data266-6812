#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    // Each 16x16 block covers a tile of C. Each thread computes one C[row, col].
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

double cpu_matmul(const std::vector<float>& A,
                  const std::vector<float>& B,
                  std::vector<float>& C,
                  int N) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

int main() {
    std::cout << "Matrix size,CPU (ms),GPU kernel (ms),H2D+D2H (ms),Speedup,Check"
              << std::endl;

    for (int N : {256, 1024, 4096}) {
        size_t total = static_cast<size_t>(N) * static_cast<size_t>(N);
        size_t bytes = total * sizeof(float);

        std::vector<float> h_A(total);
        std::vector<float> h_B(total);
        std::vector<float> h_C_cpu(total, 0.0f);
        std::vector<float> h_C_gpu(total, 0.0f);

        for (size_t i = 0; i < total; ++i) {
            h_A[i] = static_cast<float>((i % 100) / 100.0f);
            h_B[i] = static_cast<float>(((i * 3) % 100) / 100.0f);
        }

        double cpu_ms = 0.0;
        cpu_ms = cpu_matmul(h_A, h_B, h_C_cpu, N);

        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));

        cudaEvent_t h2d_start, h2d_stop;
        cudaEvent_t kernel_start, kernel_stop;
        cudaEvent_t d2h_start, d2h_stop;

        CHECK_CUDA(cudaEventCreate(&h2d_start));
        CHECK_CUDA(cudaEventCreate(&h2d_stop));
        CHECK_CUDA(cudaEventCreate(&kernel_start));
        CHECK_CUDA(cudaEventCreate(&kernel_stop));
        CHECK_CUDA(cudaEventCreate(&d2h_start));
        CHECK_CUDA(cudaEventCreate(&d2h_stop));

        CHECK_CUDA(cudaEventRecord(h2d_start));
        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(h2d_stop));
        CHECK_CUDA(cudaEventSynchronize(h2d_stop));

        // 256 threads per block is a common simple choice. The 2D grid has
        // enough blocks to cover every row and column, including edge tiles.
        dim3 threads(16, 16);
        dim3 blocks((N + threads.x - 1) / threads.x,
                    (N + threads.y - 1) / threads.y);

        CHECK_CUDA(cudaEventRecord(kernel_start));
        matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(kernel_stop));
        CHECK_CUDA(cudaEventSynchronize(kernel_stop));

        CHECK_CUDA(cudaEventRecord(d2h_start));
        CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(d2h_stop));
        CHECK_CUDA(cudaEventSynchronize(d2h_stop));

        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

        double transfer_ms = static_cast<double>(h2d_ms + d2h_ms);
        double gpu_total_ms = static_cast<double>(kernel_ms) + transfer_ms;
        double speedup = cpu_ms / gpu_total_ms;
        double max_error = 0.0;

        for (size_t i = 0; i < total; ++i) {
            double diff = std::fabs(static_cast<double>(h_C_cpu[i] - h_C_gpu[i]));
            if (diff > max_error) {
                max_error = diff;
            }
        }

        std::string check = "max error " + std::to_string(max_error);

        std::cout << N << ","
                  << std::fixed << std::setprecision(3)
                  << cpu_ms << ","
                  << kernel_ms << ","
                  << transfer_ms << ","
                  << speedup << ","
                  << check;

        std::cout << std::endl;

        CHECK_CUDA(cudaEventDestroy(h2d_start));
        CHECK_CUDA(cudaEventDestroy(h2d_stop));
        CHECK_CUDA(cudaEventDestroy(kernel_start));
        CHECK_CUDA(cudaEventDestroy(kernel_stop));
        CHECK_CUDA(cudaEventDestroy(d2h_start));
        CHECK_CUDA(cudaEventDestroy(d2h_stop));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }

    return 0;
}
