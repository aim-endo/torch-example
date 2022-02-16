cmake_minimum_required(VERSION 3.10)

if(NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "/usr/bin/clang++")
    set(CMAKE_C_LINKER "/usr/bin/clang++")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
    set(CMAKE_CXX_LINKER "/usr/bin/clang++")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("torch_example" VERSION 0.0.1 LANGUAGES CXX)

find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

set(PROJECT_INCLUDES 
    ${CMAKE_CURRENT_SOURCE_DIR}/**/*.h
    ${TORCH_INCLUDE_DIR}
)
file(GLOB_RECURSE PROJECT_SOURCES 
    main.cpp
)
set(PROJECT_LIBRARIES
    Threads::Threads
    ${TORCH_LIBRARIES}
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_INCLUDES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIBRARIES})

