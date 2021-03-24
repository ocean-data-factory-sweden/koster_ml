# Koster Object Detection Dockerfile
# author: Jannes Germishuys

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.12-py3

MAINTAINER Jannes Germishuys jurie.germishuys@combine.se

#CUDA requirements
ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update &&\
    apt-get -y upgrade &&\
    apt-get -y install build-essential cmake unzip pkg-config &&\
    apt-get -y install libjpeg-dev libpng-dev libtiff-dev &&\
    apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev &&\
    apt-get -y install libxvidcore-dev libx264-dev &&\
    apt-get -y install libgtk-3-dev &&\
    apt-get -y install libcanberra-gtk* &&\
    apt-get -y install libatlas-base-dev gfortran &&\
    apt-get -y install screen libgl1-mesa-glx &&\
    apt-get -y install python3.8-dev &&\

# Set new environment reference
ENV PYTHONPATH "/usr/local/lib/python3.8/site-packages/cv2/python-3.8/:${PYTHONPATH}"
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
RUN pip install --upgrade opencv-python
WORKDIR /usr/src/app/koster_ml
