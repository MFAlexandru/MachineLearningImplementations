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
include external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/compiler_depend.make

# Include the progress variables for this target.
include external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/progress.make

# Include the compile flags for this target's objects.
include external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/flags.make

external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/flags.make
external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: ../external/eigen-3.4.0/doc/examples/QuickStart_example.cpp
external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o: external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o -MF CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o.d -o CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o -c /home/alex/MachineLearningImplementations/external/eigen-3.4.0/doc/examples/QuickStart_example.cpp

external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alex/MachineLearningImplementations/external/eigen-3.4.0/doc/examples/QuickStart_example.cpp > CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.i

external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alex/MachineLearningImplementations/external/eigen-3.4.0/doc/examples/QuickStart_example.cpp -o CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.s

# Object files for target QuickStart_example
QuickStart_example_OBJECTS = \
"CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o"

# External object files for target QuickStart_example
QuickStart_example_EXTERNAL_OBJECTS =

external/eigen-3.4.0/doc/examples/QuickStart_example: external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/QuickStart_example.cpp.o
external/eigen-3.4.0/doc/examples/QuickStart_example: external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/build.make
external/eigen-3.4.0/doc/examples/QuickStart_example: external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alex/MachineLearningImplementations/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable QuickStart_example"
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/QuickStart_example.dir/link.txt --verbose=$(VERBOSE)
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && ./QuickStart_example >/home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples/QuickStart_example.out

# Rule to build all files generated by this target.
external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/build: external/eigen-3.4.0/doc/examples/QuickStart_example
.PHONY : external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/build

external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/clean:
	cd /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/QuickStart_example.dir/cmake_clean.cmake
.PHONY : external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/clean

external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/depend:
	cd /home/alex/MachineLearningImplementations/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alex/MachineLearningImplementations /home/alex/MachineLearningImplementations/external/eigen-3.4.0/doc/examples /home/alex/MachineLearningImplementations/build /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples /home/alex/MachineLearningImplementations/build/external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : external/eigen-3.4.0/doc/examples/CMakeFiles/QuickStart_example.dir/depend

