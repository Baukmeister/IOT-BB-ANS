# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app
COPY ./mosquitto_apps/leader .

ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116


CMD [ "python3" , "leader.py", "household", "host.docker.internal"]