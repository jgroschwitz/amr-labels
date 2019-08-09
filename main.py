### author: Jonas Groschwitz


import shutil
import tempfile
import numpy as np

from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.predictors import SentenceTaggerPredictor
from allenCode.tagger import PosDatasetReader
import allenCode.jonas_trainer # TODO this is just for registering the trainer, find a better way to do that!


# In practice you'd probably do this from the command line:
#   $ allennlp train tutorials/tagger/experiment.jsonnet -s /tmp/serialization_dir --include-package tutorials.tagger.config_allennlp
#
if __name__ == "__main__":
    params = Params.from_file('data_formatting/experiment.jsonnet')
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    # Make predictions
    predictor = SentenceTaggerPredictor(model, dataset_reader=PosDatasetReader())
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    print(tag_logits)
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    shutil.rmtree(serialization_dir)