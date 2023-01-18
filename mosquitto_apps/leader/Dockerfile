# syntax=docker/dockerfile:1

FROM python:3.8

WORKDIR /app
COPY ./mosquitto_apps/leader .
COPY ../../util ./util
COPY ./data ./data
COPY ./compression ./compression
COPY ./models ./models
COPY ./benchmark_compression.py ./benchmark_compression.py
COPY ./params/mosquitto_params ./params



ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt

CMD [ "python3" , "leader.py", "./params/household.json", "neural", "host.docker.internal"]