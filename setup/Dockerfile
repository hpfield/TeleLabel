FROM nvidia/cuda:12.3.2-devel-ubuntu22.04 

RUN apt update && apt install  openssh-server sudo -y

RUN apt update && apt install  git  -y

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 ubuntu

RUN  echo 'ubuntu:ubuntu' | chpasswd

RUN service ssh start

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]


RUN set -eux; \
        apt-get update; \
        apt-get install -y  \
                python3 \
                python3-pip \
                python3-venv \
	; \
        rm -rf /var/lib/apt/lists/*



WORKDIR /home/ubuntu
