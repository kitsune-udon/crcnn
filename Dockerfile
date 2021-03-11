FROM idein/pytorch:latest
RUN apt-get update -y
RUN pip install -U pip
RUN pip install -r requirements.txt
