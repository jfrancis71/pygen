FROM ubuntu
RUN apt update
RUN apt install -y python3-pip
RUN pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124 --break-system-packages
RUN pip install --break-system-packages tensorboard ignite
WORKDIR /home/ubuntu
ENV PYTHONPATH $PYTHONPATH:/home/ubuntu/
RUN apt install -y git
RUN apt install -y vim
RUN pip install --break-system-packages pytorch-ignite
RUN pip install --break-system-packages matplotlib

#docker run -it --rm --network=host --ipc=host -v /home/julian/datasets:/datasets -v tensorboard_logs:/tensorboard_logs -v pygen:/home/ubuntu -v $HOME/.gitconfig:/root/.gitconfig -v $HOME/.git-credentials:/root/.git-credentials --gpus=all ignite
