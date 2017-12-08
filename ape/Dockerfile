FROM pytorch_cuda9
#FROM pytorch

# torchtext 需要
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# 調整時區
ENV TZ=Asia/Taipei
RUN apt-get -y install tzdata && \
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /opt/ape
ADD . /opt/ape
RUN pip install -r requirements.txt

WORKDIR /workspace
RUN chmod -R a+w /workspace