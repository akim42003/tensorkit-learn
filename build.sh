#!/bin/bash

# Build script for tensorkit-learn C++ components and Python bindings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building tensorkit-learn C++ components and Python bindings...${NC}"

# Get the project root directory
PROJECT_ROOT="$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Navigate to bindings directory
cd "$PROJECT_ROOT/bindings"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${BLUE}Creating build directory...${NC}"
    mkdir build
fi

cd build

# Run CMake
echo -e "${BLUE}Configuring with CMake...${NC}"
if [ -n "$VIRTUAL_ENV" ]; then
    cmake .. -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python"
else
    cmake ..
fi

# Build with make using all available cores
echo -e "${BLUE}Building with make...${NC}"
make -j$(nproc)

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}Python modules installed to: $(python3 -c 'import site; print(site.getsitepackages()[0])')${NC}"