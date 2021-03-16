FROM idein/pytorch:latest
RUN apt-get update -y
RUN pip install -U pip
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
