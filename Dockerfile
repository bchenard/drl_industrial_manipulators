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

# Avoid warnings by switching to noninteractive for the build process
ENV DEBIAN_FRONTEND=noninteractive

ENV USER=root

# Install XFCE, VNC server, dbus-x11, and xfonts-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    xfce4 \
    xfce4-goodies \
    tightvncserver \
    dbus-x11 \
    xfonts-base \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Setup VNC server
RUN mkdir /root/.vnc \
    && echo "password" | vncpasswd -f > /root/.vnc/passwd \
    && chmod 600 /root/.vnc/passwd

# Create an .Xauthority file
RUN touch /root/.Xauthority

# Set display resolution (change as needed)
ENV RESOLUTION=1920x1080

# Expose VNC port
EXPOSE 5901

WORKDIR /home

COPY . /home

RUN chmod +x start_vnc.sh

CMD ["/bin/bash", "/home/entrypoint.sh"]