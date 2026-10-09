# !/bin/bash

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"

rm -rf build bin

mkdir build
mkdir bin
cd build
cmake ..
make

cp -f bin/* ../bin/
cd ..
rm -rf build