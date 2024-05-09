FROM python:3.11-slim

RUN pip install poetry==1.8.2

WORKDIR /usr/src/app

RUN pip install --target . awslambdaric

ARG EVALUATOR

COPY pyproject.toml poetry.lock poetry.toml .
COPY langevals_core/ langevals_core/
COPY evaluators/langevals evaluators/langevals
RUN poetry install --only main

COPY evaluators/$EVALUATOR evaluators/$EVALUATOR

RUN poetry install --only main --extras=$EVALUATOR
COPY langevals/ langevals/
RUN PYTHONPATH="." poetry run python langevals/server.py --preload

COPY . .

ENTRYPOINT [ "/usr/local/bin/poetry", "run", "python", "-m", "awslambdaric" ]
CMD [ "langevals.server.handler" ]
