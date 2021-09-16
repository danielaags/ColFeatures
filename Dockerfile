FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y python3-opencv

RUN pip install opencv-python
RUN pip install scikit-image
RUN pip install pandas

