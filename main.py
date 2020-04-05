### author: Jonas Groschwitz

import comet_ml # must import this before a bunch of the others, so I'll just do it first

import argparse

import shutil
import tempfile
import numpy as np

import torch
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.predictors import SentenceTaggerPredictor
import allenCode.tagger
import allenCode.jonas_trainer # TODO this is just for registering the trainer, find a better way to do that!
import allenCode.dataset_readers.amconll_aligned_lex   # just to register the reader


torch.manual_seed(0)

# In practice you'd probably do this from the command line:
#   $ allennlp train tutorials/tagger/experiment.jsonnet -s /tmp/serialization_dir --include-package tutorials.tagger.config_allennlp
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training label predictor')
    parser.add_argument('--gpu', dest="gpu", type=int, default=-1,
                        help='which gpu core to use (default -1, i.e. CPU)')
    parser.add_argument('-e', dest='experiment_name', type=str,
                        default="amr-labels-experimentation",
                        help='comet.ml experiment name')
    parser.add_argument('--noComet', dest='no_comet',
                        action='store_true',
                        help='if set, no commet.ml logging will occur (speeds up debugging and declutters comet')
    parser.add_argument('-c', dest='config_file', type=str,
                        default='data_formatting/bilstm.jsonnet',
                        help='path to config file')

    args = parser.parse_args()

    params = Params.from_file(args.config_file)
    serialization_dir = tempfile.mkdtemp()

    params.get("trainer").__setitem__("cuda_device", args.gpu)

    if args.no_comet:
        params.get("trainer").__setitem__("comet_experiment_name", None)
    else:
        params.get("trainer").__setitem__("comet_experiment_name", args.experiment_name)

    model = train_model(params, serialization_dir)

    # Make predictions
    # predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    # tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    # print(tag_logits)
    # tag_ids = np.argmax(tag_logits, axis=-1)
    # print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    shutil.rmtree(serialization_dir)