FROM python:3.11-slim

RUN pip install poetry

WORKDIR /usr/src/app

RUN pip install --target . awslambdaric

ARG EVALUATOR

COPY . .
RUN poetry install --extras=$EVALUATOR

ENTRYPOINT [ "/usr/local/bin/poetry", "run", "python", "-m", "awslambdaric" ]
CMD [ "langevals.server.handler" ]