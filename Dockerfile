FROM python:3.8.10-slim

RUN apt-get update; apt-get install git wget -y;
RUN git config --global user.name "John Doe"; git config --global user.email "johndoe@example.com";

RUN git clone https://github.com/andreybabynin/PromProdProject.git
WORKDIR /PromProdProject
RUN python3 -m pip install -r requirments.txt

RUN dvc repro

EXPOSE 5000

CMD [ "bash" ]