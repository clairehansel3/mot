cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(sample VERSION 0.0.0 LANGUAGES CXX)

add_library(sample SHARED
  src/sample.cxx
)

set_target_properties(sample PROPERTIES CXX_EXTENSIONS OFF CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_options(sample PRIVATE -O3 -ffast-math)
install(TARGETS sample DESTINATION lib)
