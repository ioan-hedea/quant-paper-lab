ARG PARENT_IMAGE=python:3.8.14-bullseye
FROM $PARENT_IMAGE

RUN apt update && \
    apt install swig xvfb freeglut3-dev ffmpeg -y 

RUN pip install gym[box2d]==0.20 torch==1.8.1 pyglet==1.5.27 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu

CMD ["/bin/bash"]
