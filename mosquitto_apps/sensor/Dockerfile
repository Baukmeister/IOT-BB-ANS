# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /app

COPY ./util ./util
COPY ./mosquitto_apps/sensor ./mosquitto_apps/sensor
COPY ./data/test_data_dfs ./data/test_data_dfs
COPY ./params/mosquitto_params ./mosquitto_apps/sensor/params
COPY ./compression/benchmark_compression.py ./compression/benchmark_compression.py

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r ./mosquitto_apps/sensor/requirements.txt

WORKDIR /app/mosquitto_apps/sensor

ENTRYPOINT [ "python3" , "sensor.py"]

