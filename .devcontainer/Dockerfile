FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt install -yq \
    python3 \
    pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libssl-dev \
    git \
    vim \
    zsh

RUN pip install --upgrade pip

RUN pip install \
    black \
    scipy \
    scikit-image \
    numba \
    torch \
    torchdata \
    torchtext \
    spacy \
    opencv-python \
    pandas \
    numpy \
    matplotlib

RUN ln -s /usr/bin/python3 /usr/bin/python

# download spacy language model for multi30k dataset.
# https://spacy.io/models
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm

# zsh install.
RUN chsh -s $(which zsh)

# oh-my-zsh setting.
RUN curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

# .zshrc 파일은 미리 플러그인이 추가된 양식을 만들어 두었다.
COPY .zshrc ~/.zshrc
