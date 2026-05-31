FROM python:3.12-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y software-properties-common python3-launchpadlib && \
    add-apt-repository ppa:jmaye/ethz && \
    apt-get update && \
    apt-get install -y  \
    build-essential liblapack-dev libblas-dev pkg-config libsuitesparse-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /bin/

ENV CPPFLAGS="-I/usr/include/suitesparse"
COPY . .

# Install runtime + dev deps into .venv. The package itself is not installed;
# tests import raise_utils from source via PYTHONPATH (set in test.sh), and the
# Cython extension is built in place below — matching the existing test flow.
RUN uv sync --frozen --no-install-project

# Put the project venv on PATH so cythonize/pytest/coverage resolve to it
ENV PATH="/usr/src/app/.venv/bin:$PATH"

# Build the Cython extension in place
WORKDIR /usr/src/app/raise_utils/transforms
RUN cythonize -i -a remove_labels.pyx

# Set the working directory to the tests directory
WORKDIR /usr/src/app/tests
ENTRYPOINT ["./test.sh"]
