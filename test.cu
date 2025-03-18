#include <iostream>
#include <cuda_runtime.h>

// カーネル関数の定義 (__global__ 修飾子を付ける)
__global__ void addKernel(int *data) {
    int idx = threadIdx.x;  // スレッドIDを取得
    data[idx] = idx;         // 各スレッドが1を加算
}

int main() {
    const int arraySize = 1024;
    int hostData[arraySize];
    for (int i ; i < arraySize ; i++){
        hostData[i] = 0;
    }
    int *deviceData;

    // GPU上にメモリを確保 (cudaMalloc)
    cudaMalloc((void**)&deviceData, arraySize * sizeof(int));
    cudaMemcpy(deviceData, hostData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // カーネルを起動 (<<<1, arraySize>>> はスレッド数を指定)
    addKernel<<<1, arraySize>>>(deviceData);
    cudaThreadSynchronize();

    // 結果をCPUにコピー (cudaMemcpy)
    cudaMemcpy(hostData, deviceData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果を表示
    int err = 0;
    for (int i = 0; i < arraySize; i++) {
        if (i != hostData[i]) {
            err +=1;
        }
    }
    std::cout << "err: " << err << std::endl;
    // std::cout << "Result[" << arraySize - 1 << "]: " << hostData[arraySize - 1] << std::endl;

    // GPUメモリを解放
    cudaFree(deviceData);

    return 0;
}