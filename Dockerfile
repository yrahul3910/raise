FROM python:3.14-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y  \
    build-essential liblapack-dev libblas-dev pkg-config libsuitesparse-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /bin/

ENV CPPFLAGS="-I/usr/include/suitesparse"
COPY . .

RUN uv sync --frozen

ENV PATH="/usr/src/app/.venv/bin:$PATH"

# Set the working directory to the tests directory
WORKDIR /usr/src/app/tests
ENTRYPOINT ["./test.sh"]
