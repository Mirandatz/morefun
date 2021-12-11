FROM python:3.10-slim-buster

COPY requirements requirements
RUN pip install -r requirements/test.txt --no-cache-dir

COPY gge gge
COPY data data

CMD pytest gge

