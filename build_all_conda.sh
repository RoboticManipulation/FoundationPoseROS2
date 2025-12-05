#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install dependencies
pip install torchvision==0.16.0+cu121 torchaudio==2.1.0 torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
# git clone https://github.com/NVlabs/FoundationPose.git

pip install gdown
pip install ruamel.yaml

git clone https://github.com/RoboticManipulation/FoundationPose.git

cd FoundationPose
# Create the weights directory and download the pretrained weights from FoundationPose
# gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37 
# gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45


## weights
if [ ! -d "weights" ] && [ ! -d "1jocuP_wFByHw6nME0ZdLDV8HVsRksZNL" ]; then
    gdown --folder  https://drive.google.com/drive/folders/1jocuP_wFByHw6nME0ZdLDV8HVsRksZNL?usp=sharing
else
    echo "Weights folder already exists, skipping download."
fi
## demo_data
if [ ! -d "demo_data" ] && [ ! -d "1PYIuQ6Q6IsF3rpqu5Hclln6Iok6qbUI0" ]; then
    gdown --folder https://drive.google.com/drive/folders/1PYIuQ6Q6IsF3rpqu5Hclln6Iok6qbUI0?usp=sharing
else
    echo "Demo data folder already exists, skipping download."
fi

# Install pybind11
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    sudo make -j6 && sudo make install

# Install Eigen
cd ${PROJ_ROOT}/FoundationPose && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    sudo make install

# Clone and install nvdiffrast
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/NVlabs/nvdiffrast && \
    cd /nvdiffrast && pip install .

# Install mycpp
cd ${PROJ_ROOT}/FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
sudo make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python3 -m pip install -e . --no-build-isolation

cd ${PROJ_ROOT}
