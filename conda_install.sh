#!/usr/bin/env bash

set -e
ENV=pytorch-pretrained-BERT

conda create -n $ENV -c pytorch pytorch==1.0 cuda90 tqdm requests boto3 -y
echo "source activate $ENV" > .env
source .env
mkdir -p lib && cd lib/ && git clone https://github.com/nvidia/apex
cd apex; python setup.py install --cuda_ext --cpp_ext; cd ..
