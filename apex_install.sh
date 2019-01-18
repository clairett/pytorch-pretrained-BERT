#!/usr/bin/env bash

set -e
mkdir -p lib && cd lib/ && git clone https://github.com/nvidia/apex
cd apex; python setup.py install --cuda_ext --cpp_ext; cd ..
