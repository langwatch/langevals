# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Argument to specify which evaluator to install
ARG EVALUATOR

# Install evaluator
RUN pip install langevals[$evaluator]

# Make port 80 available to the world outside this container
EXPOSE 80

# Run langevals-server when the container launches
CMD ["langevals-server"]