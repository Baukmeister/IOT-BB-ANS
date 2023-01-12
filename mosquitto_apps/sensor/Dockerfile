# syntax=docker/dockerfile:1

FROM python:latest
EXPOSE 1883

WORKDIR /app

COPY /mosquitto_apps/sensor/requirements.txt requirements.txt


COPY ../../util/DataLoadersLite ./util/DataLoadersLite
COPY ./mosquitto_apps/sensor .
COPY ./data ./data

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt

CMD [ "python3" , "sensor.py", "household", "data/household_power_consumption", "host.docker.internal"]

