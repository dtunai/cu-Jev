cmake_minimum_required(VERSION 3.24)
# Must be set before project(): the tensor-core kernels need sm_80 or newer.
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures (e.g. 80;86;89;90;120)")
project(cujev LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(cujev SHARED
  src/core/util.c
  src/core/config.c
  src/core/safetensors.c
  src/engine/model.c
  src/engine/forward.c
  src/api/cujev.c
  src/kernels/elementwise.cu
  src/kernels/attention.cu
  src/kernels/gdn.cu
  third_party/cJSON/cJSON.c
)
target_include_directories(cujev
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cJSON)
target_link_libraries(cujev PRIVATE CUDA::cudart CUDA::cublas m)
target_compile_options(cujev PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -O2 -fPIC>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-fPIC>)
set_target_properties(cujev PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

option(CUJEV_BUILD_TOOLS "Build the cujev-bench executable" ON)
if(CUJEV_BUILD_TOOLS)
  add_executable(cujev-bench tools/bench.c)
  target_link_libraries(cujev-bench PRIVATE cujev m)
endif()

if(SKBUILD)
  # `uv sync` / `pip install .`: the shared library lives inside the Python package
  install(TARGETS cujev LIBRARY DESTINATION cujev)
else()
  install(TARGETS cujev LIBRARY DESTINATION lib)
  install(FILES include/cujev/cujev.h DESTINATION include/cujev)
endif()
