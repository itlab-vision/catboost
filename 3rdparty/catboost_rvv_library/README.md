# Catboost RVV Library

## Directory structure

The library of CatBoost functions optimized for the RISC-V platform. The implementation provides speedup of some ways of applying CatBoost.

* `catboost_rvv_lib` is a directory that contains source codes of CatBoost functions optimized
  for the RISC-V platform (RVV 0.7.1 support).
* `main{FunctionName}` is a directory of the performance and correctness test for the `{FunctionName}` function.
* `test_data` is a directory which contains dumps obtained after enabling profiling (please, find [here](../../README_RVV.md#how-to-enable-profiling-of-computationally-intensive-functions), how to enable profiling).

## How to build

```bash
mkdir build 
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../riscv64-071-gcc.toolchain.cmake ../
make -j
```
