#!/bin/bash

# Start Xvfb
Xvfb :1 -screen 0 1024x768x16 &

# Start fluxbox
fluxbox &

# Start x11vnc
x11vnc -display :1 -forever -usepw -create