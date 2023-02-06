FROM python:3.8.16-slim-bullseye

# Update and install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install git
RUN apt-get install -y ffmpeg


# Setup environment
# RUN git clone https://github.com/Kurokabe/SceneDataset.git
WORKDIR /SceneDataset
RUN pip install poetry

COPY . .

RUN poetry install