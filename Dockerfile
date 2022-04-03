FROM python:3.8.10-slim

RUN apt-get update; apt-get install git wget -y;
COPY ./ project/
RUN python3 -m pip install -r project/requirments.txt
WORKDIR /project
RUN dvc repro
RUN python3 app.py

EXPOSE 5000

ENTRYPOINT [ "bash" ]