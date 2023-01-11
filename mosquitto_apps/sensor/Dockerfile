# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt


COPY ../../util/ ./util
COPY ./mosquitto_apps/sensor .
RUN pip3 install -r requirements.txt



CMD [ "python3" , "sensor.py", "household", "../../data/household_power_consumption"]
CMD [ "python3" , "sensor.py", "household", "../../data/household_power_consumption"]