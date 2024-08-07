FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y software-properties-common python3-launchpadlib && \
    add-apt-repository ppa:jmaye/ethz && \
    apt-get update && \
    apt-get install -y  \
    build-essential liblapack-dev libblas-dev pkg-config libsuitesparse-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPPFLAGS="-I/usr/include/suitesparse"
COPY . .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt
RUN pip install --no-cache-dir tensorflow

WORKDIR /usr/src/app/raise_utils/transforms
RUN cythonize -i -a remove_labels.pyx

# Set the working directory to the tests directory
WORKDIR /usr/src/app/tests
ENTRYPOINT ["./test.sh"]
