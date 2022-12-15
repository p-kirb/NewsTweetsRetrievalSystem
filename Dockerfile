# Description: Dockerfile for the web app
# Use ubuntu:foclal as base image
FROM ubuntu:focal  

# Set python and lucene version for docker build
ARG PYTHON_VERSION=3.9
ARG PYLUCENE_VERSION=8.11.0

# Set environment variables
# Uncomment to install specific version of poetry
ENV LANG=C.UTF-8

# ADD Python PPA Repository to ubuntu
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get remove -y software-properties-common && \
    apt-get purge --auto-remove -y && \
    apt-get clean

RUN which gpg-agent

# Install Python
RUN apt-get install -y --no-install-recommends \
        "python$PYTHON_VERSION-dev" \
        python3-setuptools \
        python3-pip && \
    apt-get remove -y gpg-agent && \
    apt-get purge --auto-remove -y && \
    apt-get clean

# ======================== START OF ADDITIONAL INSTALLATION ========================

# Install Java
RUN apt-get install -y --no-install-recommends \
    build-essential \
    ant \
    jcc \
    curl \
    git \
    default-jdk 

# Check python versions, set python3.9 as default
RUN ls /usr/bin/ | grep "python"
RUN ln -s $(which python3.9) /usr/bin/python

# Check if python set properly
RUN which python3.9 && which python && python --version

# symlink for java
WORKDIR /usr/lib/jvm/default-java/jre/lib
RUN ln -s ../../lib amd64

# check Java version
RUN java --version && javac --version


# Installing PyLucene
# Check if ant is installed (package ant for java)
RUN which ant && ant -version

# Install additional dependencies
RUN apt-get install -y --no-install-recommends \
    libffi-dev \
    zlib1g-dev

# PYLUCENE INSTALLATION
# Go to dir for pylucene
WORKDIR /usr/src/pylucene
# Download pylucene and extract
RUN curl https://dlcdn.apache.org/lucene/pylucene/pylucene-$PYLUCENE_VERSION-src.tar.gz | tar -xz
# Remove setup.py from jcc
RUN rm /usr/src/pylucene/pylucene-$PYLUCENE_VERSION/jcc/setup.py
# Copy setup.py from retrival_system (handmade)
COPY ./retrival_system/tmp/setup.py /usr/src/pylucene/pylucene-$PYLUCENE_VERSION/jcc

# Set environment variables
ENV PREFIX_PYTHON=/usr \
    JCC_JDK=/usr/lib/jvm/default-java \
    ANT=ant \
    JCC='python -m jcc' \
    NUM_FILES=10 \
    PYTHON=python \
    NO_SHARED=1

# Install PyLucene
RUN cd "pylucene-$PYLUCENE_VERSION/lucene-java-$PYLUCENE_VERSION/lucene" && \
    ant ivy-bootstrap && \
    ant && \
    cd ../../../

# Install JCC
RUN cd "pylucene-$PYLUCENE_VERSION/jcc" && \
    ls -la && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py build && \
    NO_SHARED=1 JCC_JDK=/usr/lib/jvm/default-java python setup.py install && \
    cd .. && \
    make JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8&& \
    make install JCC="python -m jcc" ANT=ant PYTHON=python NUM_FILES=8 && \
    cd ../../

# Remove unnecessary packages
RUN apt-get remove -y gpg-agent ant jcc build-essential && \
    apt-get purge --auto-remove -y && \
    apt-get clean

# Remove pylucene
WORKDIR /usr/src
RUN rm -rf pylucene


# ======================== END OF ADDITIONAL INSTALLATION ========================
# Copy data and build index
# Copy data from retrival_system/ to /retrival_system
COPY ./retrival_system /retrival_system

WORKDIR /retrival_system/lucene_system

# Install requirements for lucene
RUN python -m pip install -r requirements.txt

# Run lucence script to build index
RUN python -c "import lucene; lucene.initVM()"
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python build_index.py

WORKDIR /retrival_system/web_app

ENV FLASK_APP main.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 5000

# Install requirements for web_app
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader all

# Run web_app
CMD ["python3", "main.py"]