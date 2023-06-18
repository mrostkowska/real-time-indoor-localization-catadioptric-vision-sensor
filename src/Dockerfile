FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

RUN echo $PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=0
ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=anaconda

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    tmux \
    openssh-server \
    rsync \
    dirmngr \
    gnupg2 \
    unzip \
    htop \
    sudo \
    libgl1-mesa-glx \
    libomp-dev \
    libopenblas-dev \
    libxkbcommon-x11-0 \
    libxkbcommon-x11-0 \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV DISPLAY :1
ENV QT_X11_NO_MITSHM 1
ENV QT_GRAPHICSSYSTEM=native
ENV QT_DEBUG_PLUGINS=1
ENV TF_GPU_ALLOCATOR=cuda_malloc_async

############ SSH ############ 
RUN addgroup --system $USERNAME && \
    adduser --disabled-password --gecos "" --ingroup $USERNAME --uid 10151 $USERNAME && \
    echo $USERNAME:$USERNAME | chpasswd && \
    usermod -a -G sudo,video $USERNAME && \
    mkdir /var/run/sshd

############ ANACONDA ############
ENV ANACONDA_VERSION 2021.05
ENV ANACONDA_MD5 25e3ebae8905450ddac0f5c93f89c467

RUN wget https://repo.continuum.io/archive/Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh -O anaconda.sh && \
    echo "${ANACONDA_MD5}  anaconda.sh" > anaconda.md5 && \
    if [ $(md5sum -c anaconda.md5 | awk '{print $2}') != "OK" ] ; then exit 1; fi && \
    mkdir -p /opt && \
    sh ./anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh anaconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chown -R $USERNAME /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/$USERNAME/.profile && \
    echo "conda activate base" >> /home/$USERNAME/.profile

RUN echo $PATH
ENV PATH=/opt/conda/bin:/opt/tensorflow/bin:${PATH}
RUN echo $PATH
USER 10151

RUN conda install -c conda-forge faiss-cpu

RUN pip install --upgrade pip
RUN python -m pip install \
six \
google-api-python-client>=1.6.7 \
google-cloud-bigquery>=0.31.0 \
kaggle>=1.3.9 \
numpy>=1.15.4 \
oauth2client>=4.1.2 \
pandas>=0.22.0 \
psutil>=5.4.3 \
py-cpuinfo>=3.3.0 \
scipy>=0.19.1 \
tensorflow-hub>=0.6.0 \
tensorflow-model-optimization>=0.2.1 \
tensorflow-datasets \
tensorflow-addons \
dataclasses \
gin-config \
tf_slim>=1.1.0 \
typing \
sentencepiece \
Cython \
matplotlib \
# opencv-python-headless \
pyyaml \ 
Pillow \
-e git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI

RUN python -m pip install tensorflow
RUN python -m pip install opencv-python-headless polarTransform
#opencv-python
RUN python -m pip install -U scikit-learn==0.24.1 pyarrow

WORKDIR /app

ADD . .

EXPOSE 22