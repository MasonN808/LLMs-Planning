# Use an official base image, e.g., Debian or Ubuntu
FROM ubuntu:latest

# Install cmake
RUN apt-get update && \
    apt-get install -y cmake g++ make python3