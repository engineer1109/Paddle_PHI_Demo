# 1. Introduction
   This is a C++ Demo Paddle PHI C++.  
   这是一个Paddle Phi C++算子库的使用Demo

# 2. Build
```
git clone git@github.com:engineer1109/Paddle_PHI_Demo.git

git lfs pull
```
Rembember to Install Git Extension LFS.  
记得安装git的lfs扩展来安装，来拉取二进制文件  
里面的libpaddle_inference.so是develop 22fe4f03f611fd6be5dd3cb291814546a6cec389编译而来  
可以认为是2.4的开发版本  

在单独编译使用PHI接口的时候需要加入原先编译的宏定义， 参考本CMake文件中的add_definitions最长一行  
-DPADDLE_WITH_AVX支持AVX, -DPADDLE_WITH_CUDA支持CUDA  
不然就会发生崩溃,这个由于结构体受宏影响引起的。  

third_party是所需要的头文件，你可以替换成自己的third_party

这里编译的libpaddle_inference.so 是GCC 9.4.0 Ubuntu 20.04环境  
开启的选项 WITH_GPU=ON WITH_TESTING=ON WITH_MKL=ON WITH_AVX=ON WITH_CUSTOM_DEVICE=ON WITH_NCCL=ON

其他需要的第三方库 GTest  cudart


