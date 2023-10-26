FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app

RUN apt-get update && apt install -y \
    libgl1-mesa-glx \
    curl  \
    libglib2.0-0 && apt-get clean

RUN apt install -y git

COPY  requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


ADD . /app
RUN chmod +x /app/docker/api/start.sh
ENTRYPOINT /app/docker/api/start.sh
