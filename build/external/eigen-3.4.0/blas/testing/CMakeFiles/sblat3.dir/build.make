# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alex/MachineLearningImplementations

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alex/MachineLearningImplementations/build

# Include any dependencies generated for this target.
include external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/compiler_depend.make

# Include the progress variables for this target.
include external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/progress.make

# Include the compile flags for this target's objects.
include external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/flags.make

external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.o: external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/flags.make
external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.o: ../external/eigen-3.4.0/blas/testing/sblat3.f
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building Fortran object external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.o"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing && /usr/bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -c /home/alex/MachineLearningImplementations/external/eigen-3.4.0/blas/testing/sblat3.f -o CMakeFiles/sblat3.dir/sblat3.f.o

external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing Fortran source to CMakeFiles/sblat3.dir/sblat3.f.i"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing && /usr/bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -E /home/alex/MachineLearningImplementations/external/eigen-3.4.0/blas/testing/sblat3.f > CMakeFiles/sblat3.dir/sblat3.f.i

external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling Fortran source to assembly CMakeFiles/sblat3.dir/sblat3.f.s"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing && /usr/bin/f95 $(Fortran_DEFINES) $(Fortran_INCLUDES) $(Fortran_FLAGS) -S /home/alex/MachineLearningImplementations/external/eigen-3.4.0/blas/testing/sblat3.f -o CMakeFiles/sblat3.dir/sblat3.f.s

# Object files for target sblat3
sblat3_OBJECTS = \
"CMakeFiles/sblat3.dir/sblat3.f.o"

# External object files for target sblat3
sblat3_EXTERNAL_OBJECTS =

external/eigen-3.4.0/blas/testing/sblat3: external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/sblat3.f.o
external/eigen-3.4.0/blas/testing/sblat3: external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/build.make
external/eigen-3.4.0/blas/testing/sblat3: external/eigen-3.4.0/blas/libeigen_blas.so
external/eigen-3.4.0/blas/testing/sblat3: external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking Fortran executable sblat3"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sblat3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/build: external/eigen-3.4.0/blas/testing/sblat3
.PHONY : external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/build

external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/clean:
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing && $(CMAKE_COMMAND) -P CMakeFiles/sblat3.dir/cmake_clean.cmake
.PHONY : external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/clean

external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/depend:
	cd /home/alex/MachineLearningImplementations/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alex/MachineLearningImplementations /home/alex/MachineLearningImplementations/external/eigen-3.4.0/blas/testing /home/alex/MachineLearningImplementations/build /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/eigen-3.4.0/blas/testing/CMakeFiles/sblat3.dir/depend

