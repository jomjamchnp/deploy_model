# STEP 1: Install base image. Optimized for Python.
FROM ubuntu:18.04

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt install nvidia-driver-396

ENV CUDA_VERSION 11.3.0
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-cudart-11-2=11.2.152-1 \
#     cuda-compat-11-2 \
#     && ln -s cuda-11.2 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-3=11.3.58-1 \
    cuda-compat-11-3 \
    && ln -s cuda-11.3 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


# RUN apt-get install -y nvidia-docker2

# Required for nvidia-docker v1
# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
#     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# ENV PATH /usr/local/cuda/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/cuda-11.2/compat:/usr/local/cuda-11.2/lib64
# COPY NGC-DL-CONTAINER-LICENSE /

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450,driver<451"

# ENV CUDNN_VERSION 8.1.1.33
ENV CUDNN_VERSION 8.2.0.53
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcudnn8=$CUDNN_VERSION-1+cuda11.2 \
#     && apt-mark hold libcudnn8 && \
#     rm -rf /var/lib/apt/lists/*

# ENV CUDA_VISIBLE_DEVICES=0,1,2

# ENV NCCL_VERSION 2.8.4
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     cuda-libraries-11-2=11.2.2-1 \
#     libnpp-11-2=11.3.2.152-1 \
#     cuda-nvtx-11-2=11.2.152-1 \
#     libcublas-11-2=11.4.1.1043-1 \
#     libcusparse-11-2=11.4.1.1152-1 \
#     libnccl2=$NCCL_VERSION-1+cuda11.2 \
#     && rm -rf /var/lib/apt/lists/*

# # apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
# RUN apt-mark hold libcublas-11-2 libnccl2


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-setuptools

# Step 2: Add requirements.txt file 
COPY requirement.txt /requirement.txt

# Step 3:  Install required pyhton dependencies from requirements file
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         apt-transport-https \
#         ca-certificates \
#         curl \
#         software-properties-common \
#         python3.7 \
#         python-dev 
       
RUN apt-get update 
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirement.txt --default-timeout=10000 future
RUN apt install libgl1-mesa-glx -y
# Step 4: Copy source code in the current directory to the container
ADD . /app

# Step 5: Set working directory to previously added app directory
WORKDIR /app

# Step 6: Expose the port Flask is running on
EXPOSE 8000

# Step 9: Run Flask
CMD ["python3", "app.py", "--host=0.0.0.0", "--port=8000"]