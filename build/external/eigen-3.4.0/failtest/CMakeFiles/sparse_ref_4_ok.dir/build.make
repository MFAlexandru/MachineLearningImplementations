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
include external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/compiler_depend.make

# Include the progress variables for this target.
include external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/progress.make

# Include the compile flags for this target's objects.
include external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/flags.make

external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o: external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/flags.make
external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o: ../external/eigen-3.4.0/failtest/sparse_ref_4.cpp
external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o: external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o -MF CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o.d -o CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o -c /home/alex/MachineLearningImplementations/external/eigen-3.4.0/failtest/sparse_ref_4.cpp

external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.i"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alex/MachineLearningImplementations/external/eigen-3.4.0/failtest/sparse_ref_4.cpp > CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.i

external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.s"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alex/MachineLearningImplementations/external/eigen-3.4.0/failtest/sparse_ref_4.cpp -o CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.s

# Object files for target sparse_ref_4_ok
sparse_ref_4_ok_OBJECTS = \
"CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o"

# External object files for target sparse_ref_4_ok
sparse_ref_4_ok_EXTERNAL_OBJECTS =

external/eigen-3.4.0/failtest/sparse_ref_4_ok: external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/sparse_ref_4.cpp.o
external/eigen-3.4.0/failtest/sparse_ref_4_ok: external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/build.make
external/eigen-3.4.0/failtest/sparse_ref_4_ok: external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sparse_ref_4_ok"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sparse_ref_4_ok.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/build: external/eigen-3.4.0/failtest/sparse_ref_4_ok
.PHONY : external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/build

external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/clean:
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest && $(CMAKE_COMMAND) -P CMakeFiles/sparse_ref_4_ok.dir/cmake_clean.cmake
.PHONY : external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/clean

external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/depend:
	cd /home/alex/MachineLearningImplementations/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alex/MachineLearningImplementations /home/alex/MachineLearningImplementations/external/eigen-3.4.0/failtest /home/alex/MachineLearningImplementations/build /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/eigen-3.4.0/failtest/CMakeFiles/sparse_ref_4_ok.dir/depend

