####################
#      Global      #
####################

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)



#####################
#      Project      #
#####################

# Project variables
set(LOCAL_PROJECT_NAME        "ML")
set(LOCAL_PROJECT_VERSION     "0.0.1")
set(LOCAL_PROJECT_DESCRIPTION "Description")

# Header files (relative to "include" directory)
set(HEADERS

)

# Source files (relative to "src" directory)
set(SOURCES
    main.cpp
)

# Compiler definitions
set(DEFINES

)

# Compiler options
set(OPTIONS

)

# Project setup
project(${LOCAL_PROJECT_NAME}
        VERSION ${LOCAL_PROJECT_VERSION}
        DESCRIPTION ${LOCAL_PROJECT_DESCRIPTION}
        LANGUAGES CXX)

add_executable(${LOCAL_PROJECT_NAME})

list(TRANSFORM HEADERS PREPEND "include/")
list(TRANSFORM SOURCES PREPEND "src/")

target_include_directories(${LOCAL_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(${LOCAL_PROJECT_NAME} PRIVATE ${SOURCES} ${HEADERS})
target_compile_definitions(${LOCAL_PROJECT_NAME} PRIVATE ${DEFINES})
target_compile_options(${LOCAL_PROJECT_NAME} PRIVATE ${OPTIONS})

# Set the path to the Eigen3 cmake directory (assuming Eigen is in external/eigen)
set(EIGEN3_CMAKE_DIR ${CMAKE_SOURCE_DIR}/external/eigen-3.4.0/cmake)

# Find Eigen using the CMake configuration files in the Eigen directory
find_package(Eigen3 REQUIRED PATHS ${EIGEN3_CMAKE_DIR})

# Link Eigen3 to your target (since Eigen is header-only, this mainly ensures Eigen3::Eigen is properly found)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

set_target_properties(${LOCAL_PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "bin")
