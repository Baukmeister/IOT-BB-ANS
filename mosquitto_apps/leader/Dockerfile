# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /app
COPY ./mosquitto_apps/leader .
COPY ../../util ./util
COPY ./data ./data
COPY ./compression ./compression
COPY ./models ./models
COPY ./benchmark_compression.py ./benchmark_compression.py


ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN pip3 install -r requirements.txt
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116


CMD [ "python3" , "leader.py", "./params/mosquitto_params/household.json", "neural", "host.docker.internal"]