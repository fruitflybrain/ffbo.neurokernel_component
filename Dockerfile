###########################################
# run using:
# nvidia-docker run --name neurokernel_component -v $(dirname `pwd`):/neurokernel_component  -v $(dirname $(dirname `pwd`))/ffbo.neuroarch:/neuroarch -it ffbo/neurokernel_component:develop sh /neurokernel_component/neurokernel_component/run_component_docker.sh
# build using : nvidia-docker build -t ffbo/neurokernel_component:develop .
# require: nvidia-driver and nvidia-docker to be installed on host
###########################################

FROM nvidia/cuda:8.0-devel

MAINTAINER "Yiyin Zhou <yiyin@ee.columbia.edu>"

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get -y update
RUN apt-get -y install linux-source build-essential linux-headers-`uname -r`
RUN apt-get -y install wget libibverbs1 libnuma1 libpmi0 libslurm29 libtorque2 libhwloc-dev git libffi-dev libssl-dev

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN adduser --disabled-password --gecos '' nk

ADD . /home/nk/neurokernel_component

# Copy over private key, and set permissions
WORKDIR /home/nk
RUN mkdir .ssh
RUN chown -R nk:nk /home/nk/.ssh
RUN chown -R nk:nk /home/nk/neurokernel_component

USER nk


RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /home/nk/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/user/local/nvidia/lib64:/user/local/nvidia/lib:\$LD_LIBRARY_PATH" >> /home/nk/.bashrc

RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O /home/nk/miniconda.sh
RUN bash /home/nk/miniconda.sh -b -p /home/nk/miniconda
RUN rm miniconda.sh

# Add additional channels to .condarc
RUN echo "channels:" >> /home/nk/.condarc
RUN echo "- https://conda.binstar.org/neurokernel/channel/ubuntu1404" >> /home/nk/.condarc
RUN echo "- defaults" >> /home/nk/.condarc
#RUN echo "export PATH=/home/nk/miniconda/bin:\$PATH" >> /home/nk/.bashrc

# Setup conda environment
ENV PATH /home/nk/miniconda/bin:$PATH
RUN conda create -n NK -y python
ENV PATH /home/nk/miniconda/envs/NK/bin:$PATH

# Install dependencies
RUN conda install -n NK --yes neurokernel_deps
# NOTE: Returns "Skipping pycuda as it is not installed."
#RUN pip uninstall --yes pycuda
RUN pip install numpy==1.14.5
RUN pip install --upgrade --ignore-installed pycuda
RUN pip install autobahn[twisted]==18.12.1 simplejson pyOpenSSL service_identity
RUN pip install configparser

#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && echo $PATH && conda create -n NK -y python
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && conda install --yes neurokernel_deps
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip uninstall --yes pycuda && yes | pip install pycuda
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip install autobahn[twisted]
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && pip install simplejson && pip install pyOpenSSL && pip install service_identity

# Clone git repositories
RUN git clone --single-branch -b feature/nk_integration https://github.com/neurokernel/neurokernel.git
RUN git clone https://github.com/neurokernel/neurodriver.git
RUN git clone --single-branch -b develop https://github.com/fruitflybrain/neuroarch.git
RUN git clone https://github.com/neurokernel/retina.git

# Setup git repositories
WORKDIR /home/nk/neurokernel

# These two commands don't seem to do anything
# were they maybe useful at a previous time?
#RUN git fetch
#RUN git checkout feature/nk_integration

RUN python setup.py develop

WORKDIR /home/nk/neurodriver
RUN python setup.py develop

WORKDIR /home/nk/retina
RUN python setup.py develop

WORKDIR /home/nk/neuroarch
RUN python setup.py develop

#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && git fetch && git checkout feature/nk_integration && python setup.py develop
#WORKDIR /home/nk/neurodriver
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && python setup.py develop
#WORKDIR /home/nk/retina
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && python setup.py develop
#WORKDIR /home/nk/neuroarch
#RUN export PATH="/home/nk/miniconda/bin:${PATH}" && source activate NK && python setup.py develop

# Clear the cache
WORKDIR /home/nk/
RUN rm -rf .cache

WORKDIR /home/nk/neurokernel_component/neurokernel_component
CMD sh run_component_docker.sh
