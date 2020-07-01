# Koster Object Detection Dockerfile
# author: Jannes Germishuys

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.02-py3

MAINTAINER Jannes Germishuys jurie.germishuys@combine.se

# Create a working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
ADD https://api.github.com/repos/ocean-data-factory-sweden/koster_ml/git/refs/heads/master version.json
RUN git clone -b master https://github.com/ocean-data-factory-sweden/koster_ml.git
WORKDIR /usr/src/app/koster_ml

RUN ls -l

#RUN cd ~/ &&\
#
#    git clone https://github.com/Itseez/opencv.git &&\
#    git clone https://github.com/Itseez/opencv_contrib.git &&\
#    cd opencv && \
#    mkdir build && \
#    cd build && \
#    cmake -D CMAKE_BUILD_TYPE=RELEASE \
#    	-D CUDA_ARCH_PTX='' \
#    	-D CUDA_ARCH_BIN=6.1 \
#    	-D BUILD_OPENCV_PYTHON3=ON \
#    	-D PYTHON3_EXECUTABLE=/opt/conda/bin/python \
#    	-D PYTHON3_INCLUDE_PATH=/opt/conda/include/python3.6m \
#    	-D PYTHON3_LIBRARIES=/opt/conda/lib/python3.6/site-packages \
#        -D WITH_CUDA=ON \
#        -D ENABLE_FAST_MATH=1 \
#        -D CUDA_FAST_MATH=1 \
#        -D WITH_CUBLAS=1 \
#        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
#        .. && \
#    make -j"$(nproc)" && \
#    make install && \
#    ldconfig && \
# # Remove the opencv folders to reduce image size
#    rm -rf ~/opencv* && \
#    echo 'ln /dev/null /dev/raw1394' >> ~/.bashrc 

# Install dependencies (pip or conda)
RUN pip install -U -r requirements.txt
RUN pip install --upgrade gsutil
RUN pip install --upgrade torchvision
WORKDIR /usr/src/app/koster_ml
