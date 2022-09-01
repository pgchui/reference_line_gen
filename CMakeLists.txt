cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 14)
project(ref_line_gen)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

# find python libraries
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PythonLibs 3.0 REQUIRED)
include_directories(${PYTHON3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

# populate matplotlib repository
include(FetchContent)
FetchContent_Declare(
    matplotlib
    GIT_REPOSITORY https://github.com/Cryoris/matplotlib-cpp.git
)
FetchContent_GetProperties(matplotlib)
if(NOT matplotlib_POPULATED)
    FetchContent_Populate(matplotlib)
endif()
include_directories(SYSTEM ${matplotlib_SOURCE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen 
    OsqpEigen::OsqpEigen 
    osqp::osqp
    ${PYTHON_LIBRARIES}
    Python3::NumPy
)