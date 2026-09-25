cmake_minimum_required(VERSION 3.18)
project(tilesight_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(TS_MODEL_DIR gpuTilingPerfHWModel/model)
nanobind_add_module(_core NB_STATIC
  ${TS_MODEL_DIR}/common/cache/cache_sim.cpp
  ${TS_MODEL_DIR}/common/cache/address_map.cpp
  ${TS_MODEL_DIR}/shader_core/shader_core.cpp
  ${TS_MODEL_DIR}/shader_slice/shader_slice.cpp
  ${TS_MODEL_DIR}/on_chip_buffer/on_chip_buffer.cpp
  ${TS_MODEL_DIR}/memory_slice/memory_slice.cpp
  ${TS_MODEL_DIR}/gpu_top/hw_view.cpp
  ${TS_MODEL_DIR}/gpu_top/gpu_top.cpp
  ${TS_MODEL_DIR}/gpu_top/bindings.cpp)
target_include_directories(_core PRIVATE ${TS_MODEL_DIR})
target_link_libraries(_core PRIVATE Threads::Threads)

# scikit-build-core installs into the package dir; plain CMake builds drop it at this project's
# own root, which doubles as the "tilesight" package root (flat layout, no python/ wrapper) -
# so `from tilesight import _core` finds it right next to gpuTilingPerfHWModel/.
if(SKBUILD)
  install(TARGETS _core LIBRARY DESTINATION tilesight)
else()
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
