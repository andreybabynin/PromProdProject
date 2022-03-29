FROM python:3.8.10-slim

COPY ./ project/
RUN python3 -m pip install -r project/requirments.txt;

EXPOSE 5000

CMD ["bash"]