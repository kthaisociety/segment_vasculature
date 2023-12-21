FROM python:3.11
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR /workspaces/segment_vasculature
#COPY requirements.txt .