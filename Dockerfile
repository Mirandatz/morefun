FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# python build deps https://devguide.python.org/setup/#build-dependencies
# + git
# + curl, to download stuff
# + graphviz, to plot models
# + libgl1, for opencv2
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
    curl git graphviz libgl1 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create user and configure pyenv
ARG UNAME
ARG UID
ARG GID
RUN groupadd --gid $GID $UNAME
RUN useradd --create-home --uid $UID --gid $GID --shell /bin/bash $UNAME
USER $UNAME
ENV PYENV_ROOT /home/$UNAME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# install pyenv
SHELL ["/bin/bash", "-c"]
RUN curl https://pyenv.run | bash

# compile python 3.10.6
RUN pyenv update && CONFIGURE_OPTS="--enable-optimizations --with-lto" pyenv install 3.10.6
RUN pyenv global 3.10.6

# create project dir and change its owner
USER root
RUN mkdir /venv && chown -R $UID:$GID /venv
USER $UNAME

# install requirements
RUN python -m venv /venv
COPY ./requirements /tmp/requirements
RUN source /venv/bin/activate && pip install -r /tmp/requirements/dev.txt --no-cache-dir

# silence tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=1
