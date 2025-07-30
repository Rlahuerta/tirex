ARG MAIN_REPO=/opt/repo

# https://hub.docker.com/_/ubuntu
FROM pytorch:2.7.1-cuda11.8-cudnn9-runtime AS main

ADD  . $MAIN_REPO
WORKDIR $MAIN_REPO

RUN pip install .[gluonts] && pip install yfinance

# EXPOSE 8888
SHELL ["/bin/bash", "-c"]
