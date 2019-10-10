FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
# https://github.com/rubenrtorrado/GVGAI_GYM.git

#PYTHON
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


# NUMPY/SCIPY
RUN apt-get install -y wget unzip git \
  && pip3 install --no-input numpy\
  && pip3 install --no-input scipy 

#PYTORCH
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl \
  && pip3 install torchvision

#GYM
RUN pip3 install --no-input gym == 0.9.4 \
  && apt-get install -y swig \
  && pip3 install --no-input box2d

#display stuff
RUN pip3 install --no-input jupyter \
  && pip3 install --no-input pillow \
  && apt-get install -y x11vnc xvfb fluxbox wmctrl \
  && pip3 install --no-input matplotlib \
  && apt-get install -y python3-tk \
  && python-opengl

CMD mkdir /root/code/notebooks
WORKDIR /root/code

#GVGAI
ENV JAVA_TOOL_OPTIONS = -Dfile.encoding = UTF8
RUN apt-get install -y openjdk-9-jdk-headless \
  && git clone https://github.com/rubenrtorrado/GVGAI_GYM.git \
  && cd GVGAI_GYM \
  && pip install -e .

#ports
EXPOSE  8888
EXPOSE  6006
EXPOSE  5901
ENV DISPLAY = :1
COPY run.sh /
COPY disp_funcs.sh /
CMD ["/run.sh", "--allow-root"]
