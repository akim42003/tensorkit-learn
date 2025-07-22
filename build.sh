#!/bin/bash

# Build script for tensorkit-learn C++ components and Python bindings

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building tensorkit-learn C++ components and Python bindings...${NC}"

# Function to check dependencies
check_dependency() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo -e "${RED}Error: $1 is not installed!${NC}"
        echo -e "${YELLOW}Please install $1 before running this script.${NC}"
        exit 1
    fi
}

# Check required dependencies
echo -e "${BLUE}Checking build dependencies...${NC}"
check_dependency "cmake"
check_dependency "make"

# Check for Python
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}Error: Python is not installed!${NC}"
    echo -e "${YELLOW}Please install Python 3.7+ before running this script.${NC}"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    echo -e "${RED}Error: Python 3.7+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"

# Get the project root directory
PROJECT_ROOT="$(dirname "$0")"

# Handle virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}Using active virtual environment: $VIRTUAL_ENV${NC}"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/venv/bin/activate"
    # Update Python command to use venv's Python
    PYTHON_CMD="python"
else
    echo -e "${YELLOW}Warning: No virtual environment found. Using system Python.${NC}"
    echo -e "${YELLOW}Consider creating a virtual environment with: python -m venv venv${NC}"
fi

# Check for required Python packages
echo -e "${BLUE}Checking Python dependencies...${NC}"
if ! $PYTHON_CMD -c "import pybind11" 2>/dev/null; then
    echo -e "${RED}Error: pybind11 is not installed!${NC}"
    echo -e "${YELLOW}Please install it with: pip install pybind11${NC}"
    exit 1
fi

# Navigate to bindings directory
cd "$PROJECT_ROOT/bindings"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${BLUE}Creating build directory...${NC}"
    mkdir build
fi

cd build

# Clean previous build if requested
if [ "$1" = "clean" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf *
fi

# Run CMake
echo -e "${BLUE}Configuring with CMake...${NC}"
# Always explicitly specify Python executable for consistency
PYTHON_EXEC=$(which $PYTHON_CMD)
echo -e "${BLUE}Using Python: $PYTHON_EXEC${NC}"
if ! cmake .. -DPython_EXECUTABLE="$PYTHON_EXEC"; then
    echo -e "${RED}CMake configuration failed!${NC}"
    echo -e "${YELLOW}Try running './build.sh clean' to clean the build directory${NC}"
    exit 1
fi

# Build with make using all available cores
echo -e "${BLUE}Building with make...${NC}"
# Cross-platform CPU core detection
if command -v nproc >/dev/null 2>&1; then
    CORES=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    # macOS
    CORES=$(sysctl -n hw.ncpu)
else
    # Fallback to single core
    echo -e "${BLUE}Warning: Could not detect CPU cores, using single core build${NC}"
    CORES=1
fi
echo -e "${BLUE}Building with $CORES core(s)...${NC}"
if ! make -j$CORES; then
    echo -e "${RED}Build failed!${NC}"
    echo -e "${YELLOW}Check the error messages above for details${NC}"
    exit 1
fi

echo -e "${GREEN}Build completed successfully!${NC}"
# Use the active Python interpreter
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi
# Show installation location
INSTALL_LOCATION=$($PYTHON_CMD -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || echo "unknown")
echo -e "${GREEN}Python modules installed to: $INSTALL_LOCATION${NC}"

# Verify the build
echo -e "${BLUE}Verifying installation...${NC}"
if $PYTHON_CMD -c "import tensor_slow, link_functions, mlp_cpp, dataloader" 2>/dev/null; then
    echo -e "${GREEN}All modules imported successfully!${NC}"
else
    echo -e "${YELLOW}Warning: Could not import all modules. This might be normal if not in the virtual environment.${NC}"
    echo -e "${YELLOW}To test, activate the virtual environment and try importing the modules.${NC}"
fi

# Show usage instructions
echo -e "\n${BLUE}Usage instructions:${NC}"
echo -e "1. Activate the virtual environment (if not already active):"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo -e "2. Run your Python scripts that use the compiled modules"
echo -e "3. To rebuild from scratch: ${YELLOW}./build.sh clean${NC}"