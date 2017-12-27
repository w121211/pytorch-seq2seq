# Docker

預先要求
    * 安裝pytorch docker

```
# build
cd /path/to/ape
sudo docker build -t ape . 

# 啟動pytorch image
cd /to/work/path
sudo nvidia-docker run -it --ipc=host --name ape -v $(pwd):/workspace ape

# 啟動tensorflow (python2.7)
sudo nvidia-docker run -it --name tf -v $(pwd):/workspace tensorflow/tensorflow:latest-gpu /bin/bash

# 恢復
sudo docker start -a -i ape

# 列出所有的containers
sudo docker ps -a

# 刪除所有的containers
sudo docker rm $(sudo docker ps -a -q)
```

# Encode dataset 

SentencePiece + [Docker](https://github.com/smizy/docker-sentencepiece) + [BPEmb](https://github.com/bheinzerling/bpemb)

```
cd /path/to/ape
sudo docker run -it --rm -v $(pwd):/workspace smizy/sentencepiece sh

$ DATA=/workspace/pytorch-seq2seq/data/lang8
$ spm_encode --model=$DATA/en.wiki.bpe.op10000.model --input_format=piece < $DATA/train.cor > $DATA/train.cor.wiki10000.bpe

# 以1 Billion Word corpus 訓練 model
$ spm_train --input=/workspace/pytorch-seq2seq/data/billion/train.30m.src \ 
     --model_prefix=billion.30m.model  \ 
     --vocab_size=8000

$ DATA=/workspace/billion
$ spm_encode --model=$DATA/billion.30m.model < $DATA/train.cor > $DATA/train.cor.sp
```




