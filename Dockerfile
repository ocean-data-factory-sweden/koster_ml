# Koster Object Detection Dockerfile
# author: Jannes Germishuys

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.02-py3

MAINTAINER Jannes Germishuys jurie.germishuys@combine.se

#CUDA requirements

RUN apt-get update &&\
    apt-get -y upgrade &&\
    apt-get -y install build-essential cmake unzip pkg-config &&\
    apt-get -y install libjpeg-dev libpng-dev libtiff-dev &&\
    apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev &&\
    apt-get -y install libxvidcore-dev libx264-dev &&\
    apt-get -y install libgtk-3-dev &&\
    apt-get -y install libcanberra-gtk* &&\
    apt-get -y install libatlas-base-dev gfortran &&\
    apt-get -y install python3-dev &&\

    cd ~ &&\
    rm -r -f opencv &&\ 
    git clone https://github.com/opencv/opencv.git &&\
    rm -r -f opencv_contrib &&\
    git clone https://github.com/Itseez/opencv_contrib.git &&\

    cd ~/opencv &&\
    mkdir build &&\
    cd build &&\
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_PTX='' \
        -D CUDA_ARCH_BIN=6.1 \
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D BUILD_TESTS=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_OPENCV_PYTHON3=ON \
        -D PYTHON3_EXECUTABLE=/opt/conda/bin/python \
        -D PYTHON3_INCLUDE_PATH=/opt/conda/include/python3.6m \
        -D PYTHON3_LIBRARIES=/opt/conda/lib/python3.6/site-packages \
        -D BUILD_EXAMPLES=OFF .. && \

    make -j"$(nproc)" && \
    make install && \
    ldconfig 

# Clean-up
RUN rm -rf ~/opencv* 

# Set new environment reference
ENV PYTHONPATH "/usr/local/lib/python3.6/site-packages/cv2/python-3.6/:${PYTHONPATH}"
ENV PYTHONPATH="${PYTHONPATH}:/usr/src/app/koster_ml/src"

# Confirm path
RUN echo $PYTHONPATH

# Create a working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
ADD https://api.github.com/repos/ocean-data-factory-sweden/koster_ml/git/refs/heads/master version.json
RUN git clone -b master https://github.com/ocean-data-factory-sweden/koster_ml.git
WORKDIR /usr/src/app/koster_ml

RUN ls -l

# Install dependencies (pip or conda)
RUN pip install -U -r requirements.txt
RUN pip install --upgrade gsutil
RUN pip install --upgrade torchvision
RUN pip install --upgrade protobuf
WORKDIR /usr/src/app/koster_ml
