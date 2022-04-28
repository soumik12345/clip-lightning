# Pull Base Image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set Working Directory
RUN mkdir /usr/src/clip
WORKDIR /usr/src/clip

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.docker /usr/src/clip/requirements.docker

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.docker

COPY . /usr/src/clip/
