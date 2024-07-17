#!/bin/bash

echo 'Updating /etc/hosts file...'
HOSTNAME=$(hostname)
echo "127.0.1.1\t$HOSTNAME" >> /etc/hosts

echo "Starting XFCE4..."
startxfce4 &

echo "Starting X11VNC server..."
x11vnc -display :0 -forever -usepw -shared -rfbport 5901 &

echo "X11VNC server started on port 5901!"

# Keep the container running
tail -f /dev/null
