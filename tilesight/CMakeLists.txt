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

set(TS_CPP_DIR python/tilesight/gpuTilingPerfHWModel/model/cpp)
nanobind_add_module(_core NB_STATIC
  ${TS_CPP_DIR}/src/engine.cpp ${TS_CPP_DIR}/src/cache.cpp ${TS_CPP_DIR}/bindings/bind.cpp)
target_include_directories(_core PRIVATE ${TS_CPP_DIR}/include)
target_link_libraries(_core PRIVATE Threads::Threads)

# scikit-build-core installs into the package dir; plain CMake builds drop it in python/tilesight
if(SKBUILD)
  install(TARGETS _core LIBRARY DESTINATION tilesight)
else()
  set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/tilesight)
endif()
