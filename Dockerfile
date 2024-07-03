FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base

ENV TZ=America/Sao_Paulo
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install system deps required to compile python, see https://devguide.python.org/setup/#build-dependencies
# also install system deps required to install pyenv (which then compiles python)
FROM base AS with_python_deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    # python compilation deps
    build-essential \
    gdb \
    lcov \
    libbz2-dev \
    libffi-dev \
    libgdbm-compat-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    lzma \
    lzma-dev \
    pkg-config \
    tk-dev \
    uuid-dev \
    zlib1g-dev \
    # pyenv installation deps
    curl \
    git \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create user
FROM with_python_deps as with_user
ARG username
ARG user_id
ARG group_id

RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    sudo \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${group_id}" "${username}" \
    && useradd --create-home --shell /bin/bash --no-log-init --uid "${user_id}" --gid "${group_id}" "${username}" \
    && usermod -aG sudo "${username}" \
    && echo "${username} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# install pyenv and compile python
FROM with_user AS with_python
ARG python_version
ENV PYENV_ROOT /home/${username}/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN curl https://pyenv.run | bash
RUN pyenv update \
    && export PYTHON_CFLAGS="-march=native" \
    && export CONFIGURE_OPTS="--enable-optimizations --with-lto" \
    && pyenv install $python_version
RUN pyenv global $python_version

# install system libs, required by visualization tools or for qol
FROM with_python AS with_system_deps
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
    bash-completion \
    git-core \
    graphviz \
    libgl1 \
    sudo \
    unzip \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create project dir
FROM with_system_deps AS python_libs
RUN mkdir -p /app && chown -R ${user_id}:${group_id} /app
COPY ./requirements /app/requirements
RUN pip install -r /app/requirements/dev.txt --no-cache-dir

FROM python_libs AS final_state

# silence tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=1

# be nice with friends and share gpu ram
ENV TF_FORCE_GPU_ALLOW_GROWTH="true"

# enable cuda lazy loading
ENV CUDA_MODULE_LOADING=LAZY

# remove silly nvidia-banner
ENTRYPOINT []

# change user
USER ${username}

