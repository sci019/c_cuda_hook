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

# mallocをLD_PRELOADでフックするコード
必要最低限のサンプルコードです。

## 実行手順
### ①フックAPIをコンパイル
```bash
gcc -fPIC -shared -o alloc_api.so alloc_api.c
# gcc -fPIC -shared -o cuda_api.so cuda_api.c
```

### ②テストコードをビルドし、実行  
```bash
gcc test.c -o test
LD_PRELOAD=./alloc_api.so ./test 
```
* nvccの場合、`-cudart shared`以外の方法でコンパイルすると、フックされない


## 実行結果
* `API名、API呼び出し時刻、アドレス、確保サイズ` がログとして出力される
```
malloc,1730529137,0x555f5c03b2a0,1024 
```

### nvccを使うとどんな状況でも-cudart sharedで強制ビルドになる方法
```bash
mkdir -p ~/.local/bin
echo '#!/bin/bash' > ~/.local/bin/nvcc
echo '/usr/local/cuda/bin/nvcc "$@" -cudart shared' >> ~/.local/bin/nvcc
chmod +x ~/.local/bin/nvcc
```
* nvccのラッパースクリプトを生成

#### テスト
```bash
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cuda-12/lib64:$LD_LIBRARY_PATH
nvcc -cudart static test.cu -o test ; ./test
```
* `nvcc -cudart shared`が強制されていることが確認できる
    * 本来、`-cudart static`であれば、絶対にフックされないが、  
        `cudaMalloc,1742300366,0x7ffcebb2dca0,4096`が確認できる



# 別の方法
## 構築方法
```bash
sudo mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real
echo '#!/bin/bash' | sudo tee /usr/local/cuda/bin/nvcc
echo '/usr/local/cuda/bin/nvcc.real "$@" -cudart shared' | sudo tee -a /usr/local/cuda/bin/nvcc
sudo chmod +x /usr/local/cuda/bin/nvcc
```
* nvccのパスを絶対パスで実行してビルドするアプリでも対応可能にする

## 戻し方
```bash
sudo rm -r tee /usr/local/cuda/bin/nvcc
sudo mv /usr/local/cuda/bin/nvcc.real /usr/local/cuda/bin/nvcc
```

