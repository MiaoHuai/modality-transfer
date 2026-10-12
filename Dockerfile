FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ADD ./pytorch-CycleGAN-and-pix2pix /workspace

WORKDIR /workspace

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libgl1-mesa-glx libglib2.0-0
RUN conda install https://anaconda.org/pytorch3d/pytorch3d/0.6.2/download/linux-64/pytorch3d-0.6.2-py38_cu113_pyt1100.tar.bz2
RUN pip install -r requirements.txt

EXPOSE 5001

CMD ["python", "server.py"]