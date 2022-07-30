cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CXX_STANDARD 14)
project(ref_line_gen)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
    Eigen3::Eigen 
    OsqpEigen::OsqpEigen 
    osqp::osqp
)