# !/bin/bash

# make sure the working directory is the same as the script
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

# clean up previous build and bin directories
rm -rf build bin

# create directories
mkdir build
mkdir bin
cd build
cmake ..
make

# copy all the binaries to the bin directory
cp -f bin/* ../bin/
# remove the build directory
cd ..
rm -rf build