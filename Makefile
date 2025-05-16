# Makefile for N-Body Simulation C++ Project (CPU Only)

# --- Compiler and Linker ---
CXX = g++
LD = g++

# --- Directories ---
SRC_DIR = src
APP_DIR = app
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
TARGET_EXEC = $(BIN_DIR)/nbody_sim

# --- Source Files (CPU Only) ---
# Add all your .cpp files here.
# Ensure paths are relative to the Makefile's location (project root).
CPP_SOURCES = \
    $(SRC_DIR)/particle_data.cpp \
    $(SRC_DIR)/initial_conditions.cpp \
    $(SRC_DIR)/backends/cpu_n2_backend.cpp \
    $(SRC_DIR)/integrator.cpp \
    $(SRC_DIR)/simulation.cpp \
    $(SRC_DIR)/main.cpp

# --- Object Files ---
# Generate object file names from source file names, placing them in OBJ_DIR
# Example: src/data_structures/ParticleData.cpp -> build/obj/src/data_structures/ParticleData.o
CPP_OBJECTS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))

# --- Compiler Flags ---
# Common flags for C++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g # -O2 for optimization, -g for debug symbols
# For a release build without debug symbols, you might use -O3 and later `strip $(TARGET_EXEC)`

# Include directories for C++
# Add -I for each directory containing header files your .cpp files need.
INCLUDE_DIRS = -I$(SRC_DIR)/backends \
               -I$(SRC_DIR) # For headers directly in src/, like ComputeBackend.h

# --- Linker Flags ---
LDFLAGS = -lm # Link math library, often needed

# --- Final Flags ---
FINAL_CXXFLAGS = $(CXXFLAGS) $(INCLUDE_DIRS)

# --- Targets ---
.PHONY: all clean run

all: $(TARGET_EXEC)

$(TARGET_EXEC): $(CPP_OBJECTS) | $(BIN_DIR) # Depends on all .o files and BIN_DIR existing
	@echo "Linking..."
	$(LD) $(CPP_OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $@"

# Pattern rule to compile .cpp files from any subdirectory under SRC_DIR or APP_DIR
# This generic rule simplifies handling various source locations.
# It assumes object files mirror the source path under $(OBJ_DIR).
$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR) # Prerequisite: corresponding .cpp file, and OBJ_DIR must exist
	@mkdir -p $(@D) # Create directory for object file if it doesn't exist ($(dir $@))
	@echo "Compiling C++: $<"
	$(CXX) $(FINAL_CXXFLAGS) -c $< -o $@

# Create build directories if they don't exist (Order-only prerequisites)
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILD_DIR)
	@echo "Clean complete."

run: all
	@echo "Running simulation..."
	$(TARGET_EXEC)