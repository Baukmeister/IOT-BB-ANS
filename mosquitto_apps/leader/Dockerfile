# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app


COPY ./mosquitto_apps/leader .
COPY ./data ./data

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt

CMD [ "python3" , "leader.py", "household", "host.docker.internal"]