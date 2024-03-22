FROM python:3.10.12-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt

# Install build-essential and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
