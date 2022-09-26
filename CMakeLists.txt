cmake_minimum_required(VERSION 3.10)

project(phi_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath='./'")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fPIC")

add_definitions(-DCUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" -DCUDA_VERSION_MAJOR="11" -DCUDA_VERSION_MINOR="7" -DCUDNN_MAJOR_VERSION="8" -DEIGEN_USE_GPU -DLAPACK_FOUND -DPADDLE_DISABLE_PROFILER -DPADDLE_DLL_EXPORT -DPADDLE_ON_INFERENCE -DPADDLE_USE_PTHREAD_BARRIER -DPADDLE_USE_PTHREAD_SPINLOCK -DPADDLE_VERSION=0.0.0 -DPADDLE_VERSION_INTEGER=0 -DPADDLE_WITH_AVX -DPADDLE_WITH_CRYPTO -DPADDLE_WITH_CUDA -DPADDLE_WITH_CUPTI -DPADDLE_WITH_CUSTOM_DEVICE -DPADDLE_WITH_MKLDNN -DPADDLE_WITH_MKLML -DPADDLE_WITH_NCCL -DPADDLE_WITH_POCKETFFT -DPADDLE_WITH_SSE3 -DPADDLE_WITH_TESTING -DPADDLE_WITH_XBYAK -DTHRUST_IGNORE_CUB_VERSION_CHECK -DTRT_PLUGIN_FP16_AVALIABLE -DXBYAK64 -DXBYAK_NO_OP_NAMES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fPIC")

find_package(CUDA)
find_package(GTest)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/third_party/paddle/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/protobuf/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/mkldnn/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${PROJECT_SOURCE_DIR}/third_party/protobuf/lib)

add_executable(main add_test.cc abs_test.cc main.cc )

target_link_libraries(main 
    paddle_inference
    protobuf
    ${GTEST_LIBRARIES} dl cudart)
