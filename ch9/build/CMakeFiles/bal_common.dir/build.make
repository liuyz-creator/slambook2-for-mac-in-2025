# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/liuyz/slam/slambook2/ch9

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/liuyz/slam/slambook2/ch9/build

# Include any dependencies generated for this target.
include CMakeFiles/bal_common.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/bal_common.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/bal_common.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bal_common.dir/flags.make

CMakeFiles/bal_common.dir/codegen:
.PHONY : CMakeFiles/bal_common.dir/codegen

CMakeFiles/bal_common.dir/common.cpp.o: CMakeFiles/bal_common.dir/flags.make
CMakeFiles/bal_common.dir/common.cpp.o: /Users/liuyz/slam/slambook2/ch9/common.cpp
CMakeFiles/bal_common.dir/common.cpp.o: CMakeFiles/bal_common.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/liuyz/slam/slambook2/ch9/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bal_common.dir/common.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/bal_common.dir/common.cpp.o -MF CMakeFiles/bal_common.dir/common.cpp.o.d -o CMakeFiles/bal_common.dir/common.cpp.o -c /Users/liuyz/slam/slambook2/ch9/common.cpp

CMakeFiles/bal_common.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/bal_common.dir/common.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/liuyz/slam/slambook2/ch9/common.cpp > CMakeFiles/bal_common.dir/common.cpp.i

CMakeFiles/bal_common.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/bal_common.dir/common.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/liuyz/slam/slambook2/ch9/common.cpp -o CMakeFiles/bal_common.dir/common.cpp.s

# Object files for target bal_common
bal_common_OBJECTS = \
"CMakeFiles/bal_common.dir/common.cpp.o"

# External object files for target bal_common
bal_common_EXTERNAL_OBJECTS =

libbal_common.a: CMakeFiles/bal_common.dir/common.cpp.o
libbal_common.a: CMakeFiles/bal_common.dir/build.make
libbal_common.a: CMakeFiles/bal_common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/liuyz/slam/slambook2/ch9/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libbal_common.a"
	$(CMAKE_COMMAND) -P CMakeFiles/bal_common.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bal_common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bal_common.dir/build: libbal_common.a
.PHONY : CMakeFiles/bal_common.dir/build

CMakeFiles/bal_common.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bal_common.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bal_common.dir/clean

CMakeFiles/bal_common.dir/depend:
	cd /Users/liuyz/slam/slambook2/ch9/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/liuyz/slam/slambook2/ch9 /Users/liuyz/slam/slambook2/ch9 /Users/liuyz/slam/slambook2/ch9/build /Users/liuyz/slam/slambook2/ch9/build /Users/liuyz/slam/slambook2/ch9/build/CMakeFiles/bal_common.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/bal_common.dir/depend

