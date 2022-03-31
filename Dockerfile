FROM python:3.8.10-slim

COPY ./ project/
RUN python3 -m pip install -r project/requirments.txt
WORKDIR /project
RUN python3 app.py
EXPOSE 5000

# CMD ["bash"]