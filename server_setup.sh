#!/bin/bash

# start conda
. /proj/contrib/anaconda3/etc/profile.d/conda.sh

# for comet.ml
export https_proxy="http://www-proxy.uni-saarland.de:3128"

# set correct temp directory (the default, /tmp/, is a bit small)
export TMPDIR=/local/jonasg/tmp/

# activate the right conda environment
conda activate /proj/irtg.shadow/jonas-conda/allennlp

