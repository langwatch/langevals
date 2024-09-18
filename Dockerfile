FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc g++ make

RUN pip install poetry==1.8.2

WORKDIR /usr/src/app

COPY pyproject.toml poetry.lock poetry.toml .
COPY langevals_core/ langevals_core/
COPY evaluators/langevals evaluators/langevals
RUN poetry install --only main

COPY evaluators evaluators

RUN poetry install --only main --all-extras
COPY langevals/ langevals/
RUN PYTHONPATH="." poetry run python langevals/server.py --preload

COPY . .

CMD PYTHONPATH="." poetry run python langevals/server.py
