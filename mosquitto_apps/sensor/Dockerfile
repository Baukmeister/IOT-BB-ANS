# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /app

COPY ../../util ./util
COPY ./mosquitto_apps/sensor .
COPY ./data/test_data_dfs ./data/test_data_dfs
COPY ./params/mosquitto_params ./params

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt


ENTRYPOINT [ "python3" , "sensor.py", "data/test_data_dfs"]

