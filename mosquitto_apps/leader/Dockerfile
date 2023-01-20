# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3

WORKDIR /app
COPY ./mosquitto_apps/leader .
COPY ../../util ./util
COPY ./compression ./compression
COPY ./models ./models
COPY ./benchmark_compression.py ./benchmark_compression.py
COPY ./params/mosquitto_params ./params



ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt


ENTRYPOINT [ "python3" , "leader.py", "neural"]