cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

project(initia VERSION 0.0)

set(SOURCE_FILES src/main.cpp)

add_subdirectory(vendor/OpenSYCL)
find_package(OpenSYCL CONFIG REQUIRED)
# find_package(wild REQUIRED) TODO: make work
add_executable(initia ${SOURCE_FILES})

add_sycl_to_target(TARGET initia)
file(GLOB_RECURSE sources src/*.cpp)
target_sources(initia PRIVATE ${sources})

target_include_directories(${PROJECT_NAME} PUBLIC "src/" "vendor/OpenSYCL/include" "vendor/eigen") # "wild/include" "wild/src/include") # hack to make assimp work
# target_link_libraries(tracer OpenCL::OpenCL)
# target_link_libraries(tame wild) # bgfx bx bimg glfw)
