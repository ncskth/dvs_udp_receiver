
cmake_minimum_required(VERSION 3.7)
project(UDPStream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("include/")

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Connect to socket
add_library(client SHARED client.cpp)

# Listening binary
add_executable(listening udpstream.cpp)
target_link_libraries(listening PRIVATE client "${TORCH_LIBRARIES}")

# UDP stream python
# pybind11_add_module(udpstream udpstream.cpp)
# target_link_libraries(udpstream PRIVATE client "${TORCH_LIBRARIES}")