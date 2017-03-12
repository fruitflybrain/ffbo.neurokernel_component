###########################################
# run using: 
# nvidia-docker run -it ffbo/neurokernel sh run_component.sh ws://processor_ip:8080/ws
# build using : nvidia-docker build -t ffbo/neurokernel .
# require: nvidia-driver and nvidia-docker to be installed on host
###########################################

FROM nvidia/cuda

LABEL maintainer "Yiyin Zhou <yiyin@ee.columbia.edu>"

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get -y update
RUN apt-get -y install linux-source build-essential linux-headers-`uname -r`
RUN apt-get -y install wget libibverbs1 libnuma1 libpmi0 libslurm29 libtorque2 libhwloc-dev git libffi-dev libssl-dev

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN adduser --disabled-password --gecos '' nk

# Copy over private key, and set permissions
WORKDIR /home/nk
RUN mkdir .ssh
RUN chown -R nk:nk /home/nk/.ssh

USER nk

RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /home/nk/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/user/local/nvidia/lib64:/user/local/nvidia/lib:\$LD_LIBRARY_PATH" >> /home/nk/.bashrc

RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /home/nk/miniconda.sh
RUN bash /home/nk/miniconda.sh -b -p /home/nk/miniconda
RUN rm miniconda.sh

RUN echo "channels:" >> /home/nk/.condarc
RUN echo "- https://conda.binstar.org/neurokernel/channel/ubuntu1404" >> /home/nk/.condarc
RUN echo "- defaults" >> /home/nk/.condarc
RUN echo "export PATH=/home/nk/miniconda/bin:\$PATH" >> /home/nk/.bashrc

RUN export PATH="/home/nk/miniconda/bin:${PATH}" && echo $PATH && conda create -n NK -y python
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && conda install --yes neurokernel_deps
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip uninstall --yes pycuda && yes | pip install pycuda
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip install autobahn[twisted]
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip install simplejson

RUN git clone https://github.com/fruitflybrain/neurokernel.git
RUN git clone https://github.com/fruitflybrain/neurodriver.git
RUN git clone https://github.com/fruitflybrain/neuroarch.git
RUN git clone https://github.com/fruitflybrain/ffbo.neurokernel_component.git
RUN git clone https://github.com/neurokernel/retina.git

WORKDIR /home/nk/neurokernel
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && git fetch && git checkout feature/nk_integration && python setup.py install
WORKDIR /home/nk/neurodriver
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && python setup.py install
WORKDIR /home/nk/retina
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && git fetch && git checkout models-abstraction && python setup.py install

WORKDIR /home/nk/neuroarch
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && git fetch && git checkout feature/nk_integration && python setup.py install

WORKDIR /home/nk/
RUN rm -rf .cache

WORKDIR /home/nk/neurokernel_component/neurokernel_component
RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && python setup.py install




