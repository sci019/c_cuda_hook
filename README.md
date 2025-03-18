# C/CUDA関数をLD_PRELOADでフックするコード
必要最低限のサンプルコードです。

## 実行手順
### ①フックAPIをコンパイル
```bash
gcc -fPIC -shared -o cuda_api.so cuda_api.c
```

### ②nvccを使うと絶対に-cudart sharedでビルドされるようにする設定
```bash
mkdir -p ~/.local/bin
echo '#!/bin/bash' > ~/.local/bin/nvcc
echo '/usr/local/cuda/bin/nvcc "$@" -cudart shared' >> ~/.local/bin/nvcc
chmod +x ~/.local/bin/nvcc
```
* nvccのラッパースクリプトを生成

### ③テストコードをビルドし、実行  
```bash
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cuda-12/lib64:$LD_LIBRARY_PATH
nvcc -cudart static test.cu -o test ; ./test
```
* `nvcc -cudart shared`が強制される

### 実行結果
* `API名、API呼び出し時刻、アドレス、確保サイズ` がログとして出力される
```
`cudaMalloc,1742300366,0x7ffcebb2dca0,4096
```
 * 本来、`-cudart static`を明示すれば、絶対にフックされないが、  
     `cudaMalloc`のフックが確認できる
