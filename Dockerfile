# Use official Ubuntu 22.04
FROM ubuntu:22.04

LABEL mantainer="gianni.lunardi@unitn.it"

# No user interaction during package installation
# ENV DEBIAN_FRONTEND=noninteractive

# My user
ARG USERNAME=gianni
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install packages from apt
RUN apt-get update && apt-get install -y python3 python3-pip cmake git build-essential 

# Install pip packages
RUN pip install --upgrade pip
RUN pip install numpy scipy matplotlib \
    tqdm PyYAML \ 
    setuptools cmake ninja scikit-build \
    casadi

# Install pytorch 
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Create the necessary directories
RUN mkdir -p /home/$USERNAME/external /home/$USERNAME/devel

# Install acados
WORKDIR /home/$USERNAME/external
RUN git clone https://github.com/acados/acados.git \
    && cd acados \
    && git submodule update --recursive --init \
    && mkdir -p build \
    && cd build \
    && cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON -DACADOS_PYTHON=ON ..  \
    && make install -j4

RUN pip install acados/interfaces/acados_template

# Install l4casadi
RUN pip install --upgrade setuptools
RUN pip install l4casadi --no-build-isolation
# RUN git clone https://github.com/Tim-Salzmann/l4casadi.git \
#     && cd l4casadi \
#     && pip install . --no-build-isolation 

# Link to shared libraries (acados)
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USERNAME/external/acados/lib" >> /home/$USERNAME/.bashrc
RUN echo "export ACADOS_SOURCE_DIR=/home/$USERNAME/external/acados" >> /home/$USERNAME/.bashrc

# Link to python packages
RUN echo "export PYTHONPATH=$PYTHONPATH:/home/$USERNAME/devel/safe-mpc/src" >> /home/$USERNAME/.bashrc

USER $USERNAME
WORKDIR /home/$USERNAME/