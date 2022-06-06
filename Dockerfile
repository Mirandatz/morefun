FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# python build deps https://devguide.python.org/setup/#build-dependencies
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# to download python :)
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    git \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# compile python 3.10.4
RUN git clone --depth 1 https://github.com/python/cpython.git --branch v3.10.4 \
    && cd /cpython \
    && ./configure --enable-optimizations \
    && make \
    && make install \
    && update-alternatives --install /usr/bin/python python /usr/local/bin/python3 999 \
    && rm -rf /cpython

WORKDIR /gge/

COPY ./requirements ./requirements
RUN python -m pip install -r ./requirements/dev.txt --no-cache-dir

ENV TF_CPP_MIN_LOG_LEVEL=1
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
