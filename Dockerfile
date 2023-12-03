FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install basic system deps
FROM base AS with_system_deps
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    curl \
    git \
    git-core \
    bash-completion \
    graphviz \
    libgl1 \
    unzip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# install python deps https://devguide.python.org/setup/#build-dependencies
FROM with_system_deps AS with_python_deps
ARG PYTHON_VERSION
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create user
FROM with_python_deps as with_user
ARG UNAME
ARG UID
ARG GID
RUN groupadd --gid $GID $UNAME
RUN useradd --create-home --uid $UID --gid $GID --shell /bin/bash $UNAME
USER $UNAME
ENV PYENV_ROOT /home/$UNAME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# install pyenv and compile python
FROM with_user AS with_python
ARG PYTHON_VERSION
ENV PYENV_ROOT /home/$UNAME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN curl https://pyenv.run | bash
RUN pyenv update \
    && PYTHON_CFLAGS="-march=native" \
    CONFIGURE_OPTS="--enable-optimizations --with-lto" \
    pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION

# create project dir and change its owner
USER root
RUN mkdir -p /app/.venv && chown -R $UID:$GID /app/.venv
USER $UNAME

# install requirements
RUN python -m venv /app/.venv
COPY ./requirements /app/requirements
RUN source /app/.venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /app/requirements/dev.txt --no-cache-dir

# silence tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=1

# enable xla
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

# be nice with friends and share gpu ram
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"

# remove silly nvidia-banner
ENTRYPOINT []

# enable cuda lazy loading
ENV CUDA_MODULE_LOADING=LAZY
