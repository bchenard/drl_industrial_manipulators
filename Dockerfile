FROM ubuntu:22.04


LABEL author="Benoît Chénard"
LABEL email="bchenard@bordeaux-inp.fr"
LABEL version="1.0"
LABEL description="DRL Industrial Manipulators Image"


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 

RUN pip3 install --upgrade pip \
    && pip3 install numpy \
    && pip3 install matplotlib \
    && pip3 install tensorflow \
    && pip3 install pybullet \
    && pip3 install gymnasium \
    && pip3 install stable-baselines3 \
    && pip3 install argparse \
    && pip3 install torch

WORKDIR /home

COPY . /home

CMD ["/bin/bash", "/home/entrypoint.sh"]