FROM python:3.14.2-slim

RUN pip install poetry==1.8.2

WORKDIR /usr/src/app

RUN pip install --target . awslambdaric

COPY pyproject.toml poetry.lock poetry.toml .
COPY langevals_core/ langevals_core/
COPY evaluators/langevals evaluators/langevals
RUN poetry install --only main

COPY evaluators evaluators

RUN poetry install --only main --all-extras
COPY langevals/ langevals/
RUN PYTHONPATH="." poetry run python langevals/server.py --preload

ENV RUNNING_IN_DOCKER=true

COPY . .

CMD PYTHONPATH="." poetry run python langevals/server.py
