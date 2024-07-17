FROM ubuntu:22.04

LABEL author="Benoît Chénard"
LABEL email="bchenard@bordeaux-inp.fr"
LABEL version="1.0"
LABEL description="DRL Industrial Manipulators Image"

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python-is-python3 \
    x11vnc \
    xvfb \
    fluxbox \
    xterm \
    wget \
    net-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables
ENV DISPLAY=:1

# Create the .vnc directory and password
RUN mkdir -p ~/.vnc && \
    x11vnc -storepasswd 1234 ~/.vnc/passwd

# Create a startup script
RUN echo '#!/bin/bash\n\
          \n\
          # Start Xvfb\n\
          Xvfb :1 -screen 0 1024x768x16 &\n\
          \n\
          # Start fluxbox\n\
          fluxbox &\n\
          \n\
          # Start x11vnc\n\
          x11vnc -display :1 -forever -usepw -create\n' > /startup.sh \
          && chmod +x /startup.sh

# Install Python libraries
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


# Expose VNC port
EXPOSE 5901

WORKDIR /home

COPY . /home

RUN chmod +x start_vnc.sh

CMD ["/bin/bash", "/home/entrypoint.sh"]
