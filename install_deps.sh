#!/usr/bin/env bash
set -e

echo "Installing base requirements..."
uv pip install -r requirements.txt

echo "Installing CuMesh..."
mkdir -p extensions
if [ ! -d "extensions/CuMesh" ]; then
    git clone https://github.com/JeffreyXiang/CuMesh.git extensions/CuMesh --recursive
fi
uv pip install extensions/CuMesh --no-build-isolation

echo "Installing FlexGEMM..."
if [ ! -d "extensions/FlexGEMM" ]; then
    git clone https://github.com/JeffreyXiang/FlexGEMM.git extensions/FlexGEMM --recursive
fi
uv pip install extensions/FlexGEMM --no-build-isolation

echo "Installing nvdiffrast..."
if [ ! -d "extensions/nvdiffrast" ]; then
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git extensions/nvdiffrast
fi
uv pip install extensions/nvdiffrast --no-build-isolation

echo "Installing o-voxel..."
uv pip install ./o-voxel --no-build-isolation

echo "All dependencies installed."
