#!/bin/bash

# start conda
. /proj/contrib/anaconda3/etc/profile.d/conda.sh

# for comet.ml
export https_proxy="http://www-proxy.uni-saarland.de:3128"

# activate the right conda environment
conda activate --prefix /proj/irtg.shadow/jonas-conda/allennlp

