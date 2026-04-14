# Use a base image with Conda
FROM continuumio/miniconda3


# Install curl, bash, and other dependencies
RUN apt-get update && apt-get install -y curl bash build-essential \
    && curl https://sh.rustup.rs -sSf | bash -s -- -y

# Set the working directory in the container
WORKDIR /app

# Ensure the Rust binaries are available in the PATH
ENV PATH="/root/.cargo/bin:${PATH}"


# Copy the environment YAML file into the Docker image
COPY medgraphrag.yml .

RUN conda config --append channels conda-forge
RUN conda config --append channels pkgs/main


# Create the Conda environment inside the Docker container
RUN conda env create -f medgraphrag.yml

# Make sure the environment is activated and Conda commands are available
RUN echo "source activate medgraphrag" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy the rest of the application code into the Docker image
COPY . /app

