# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app

COPY ../../util ./util
COPY ./mosquitto_apps/sensor .
COPY ./data ./data
COPY ./params/mosquitto_params ./params

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt

CMD [ "python3" , "sensor.py", "data/household_power_consumption", "params/household.json", "host.docker.internal"]

