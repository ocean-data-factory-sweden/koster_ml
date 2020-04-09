# Koster Object Detection Dockerfile
# author: Jannes Germishuys

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.02-py3

MAINTAINER Jannes Germishuys jurie.germishuys@combine.se

# Create a working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
RUN git clone https://github.com/ocean-data-factory-sweden/koster_ml
WORKDIR /usr/src/app/koster_ml
RUN ls -l

# Install dependencies (pip or conda)
RUN pip install -U gsutil
RUN pip install -U -r requirements.txt
RUN pip install --upgrade torchvision
RUN conda update -n base -c defaults conda
RUN conda install -y -c anaconda future numpy opencv matplotlib tqdm pillow
RUN conda install -y -c conda-forge scikit-image tensorboard pycocotools

# Copy weights
WORKDIR /usr/src/app/koster_ml
RUN python3 -c "from models import *; \
attempt_download('weights/yolov3.pt'); \
attempt_download('weights/darknet53.conv.74'); \
attempt_download('weights/yolov3-spp.pt')"