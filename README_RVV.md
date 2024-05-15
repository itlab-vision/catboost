# How to build Catboost with RVV 0.7.1 support

The library is built in two stages:

* Building the library of CatBoost functions optimized for a RISC-V platform using a cross-compiler (Catboost RVV library).
* Building the Сatboost library using a native compiler on a RISC-V platform.

## How to build the library optimized for RISC-V

It is required a cross-compiler with the vector extension RVV 0.7.1.

The library is compiled using the following instructions:

```bash
cur_dir=`pwd`
cd 3rdparty/catboost_rvv_library
mkdir build 
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../riscv64-071-gcc.toolchain.cmake ../
make -j
cd $cur_dir
```

# How to build the Сatboost library

To install dependencies, please, use the following instructions.

```bash
apt-get update
apt install clang
apt install cmake
apt install ninja-build
apt install lld-15  
apt install openssl
apt install libssl-dev
```

To compile the CatBoost library on a RISC-V platform, please, follow instructions below.

```bash
catboost_source_dir=<path to the library>
catboost_build_dir="$catboost_source_dir/build_no_cuda"
cur_dir=`pwd`

echo "creating a virtual environment ..."
mkdir $cur_dir/py_venv
python -m venv $cur_dir/py_venv/py39
source $cur_dir/py_venv/py39/bin/activate
  
echo "install dependencies ..."
pip install -U pip setuptools
pip install six
pip install wheel
pip install numpy
pip install pandas
pip install scipy

echo "build catboost for riscv64 ..."
python $catboost_source_dir/build/build_native.py --build-root-dir=$catboost_build_dir --targets _catboost --verbose

echo "build catboost python package for riscv64 ..."
cd $catboost_source_dir/catboost/python-package/
python setup.py bdist_wheel --prebuilt-extensions-build-root-dir=$catboost_source_dir/build_no_cuda --no-widget
cd $cur_dir

echo "install catboost python package for riscv64 ..."
pip install $catboost_source_dir/catboost/python-package/dist/catboost-1.2.2-cp39-cp39-linux_riscv64.whl --no-deps
```

## How to enable profiling of computationally intensive functions

To optimize applying the trained CatBoost model, a test dataset is required. To check correctness
of the algorithm after each optimization step you should dump intermediate results
for each function of the call stack. To enable saving intermediate results, please,
add one of the following definitions in the `catboost/time_profile.h` file.

```cpp
#define ___DUMP_CalculateLeafValues
#define ___DUMP_CalculateLeafValuesMulti
#define ___DUMP_BinarizeFloats
#define ___DUMP_CalcIndexesBasic_XOR
#define ___DUMP_CalcIndexesBasic
```

You can enable profiling of the computationally intensive functions by adding the definition
below to the `catboost/time_profile.h` file. It allows to determine the number
of function calls and calculation time.

```cpp
#define __TIME_PROF___
```

You can enable profiling of computationally intensive functions to determine calculation time
only by adding the definition below to the `catboost/time_profile.h` file (this approach has
less impact on overall calculation time).

```cpp
#define __TIME_PROF_2___
```

## How to run performance tests

```bash
cd samples/training_benchmarking_rvv_library
python main_benchmark.py
```
