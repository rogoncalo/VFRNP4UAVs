# Use Python 3.9 slim base image
FROM python:3.9-slim

# Install essential build tools and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libx11-xcb-dev \
        wget \
        xvfb \
        xauth && \
    rm -rf /var/lib/apt/lists/*

# Set PATH to include Miniconda binaries
ENV PATH="/opt/conda/bin:${PATH}"

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Initialize Conda and set up the base environment
RUN /opt/conda/bin/conda init bash
RUN echo "conda activate base" >> ~/.bashrc

# Create and activate a new Conda environment
RUN /opt/conda/bin/conda create -n drones python=3.9
RUN echo "conda activate drones" >> ~/.bashrc

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install the gym_pybullet_drones package
RUN pip install .

# Set display environment variable for Xvfb
ENV DISPLAY=:99

# Specify the command to run on container start
CMD ["xvfb-run", "-s", ":99", "python", "gym_pybullet_drones/examples/pid.py"]
