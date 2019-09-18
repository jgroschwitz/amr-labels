### based on https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#using-config-files

from typing import Iterator, List, Dict

import torch

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

from allenCode.f_metric import MultisetFScore

torch.manual_seed(1)

@DatasetReader.register('pos-tutorial')
class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)



@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2label1 = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.hidden2label2 = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.hidden2label3 = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy1 = CategoricalAccuracy()
        self.accuracy2 = CategoricalAccuracy()
        self.accuracy3 = CategoricalAccuracy()
        self.fscore = MultisetFScore(null_label_id=vocab.get_token_index(token="_", namespace='labels'))
        self.perform_expensive_eval = True


    def set_perform_expensive_eval(self, do_it:bool):
        self.perform_expensive_eval = do_it

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels1: torch.LongTensor = None,
                labels2: torch.LongTensor = None,
                labels3: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        label1_logits = self.hidden2label1(encoder_out)
        label2_logits = self.hidden2label2(encoder_out)
        label3_logits = self.hidden2label3(encoder_out)
        output = {"label1_logits": label1_logits, "label2_logits": label2_logits, "label3_logits": label3_logits}
        if labels1 is not None and labels2 is not None and labels3 is not None:
            self.accuracy1(label1_logits, labels1, mask)
            self.accuracy2(label2_logits, labels2, mask)
            self.accuracy3(label3_logits, labels3, mask)
            if self.perform_expensive_eval:
                # otherwise we don't count things for fscore, and it will just be 0.
                self.fscore([label1_logits, label2_logits, label3_logits], [labels1, labels2, labels3], mask)
            output["loss"] = sequence_cross_entropy_with_logits(label1_logits, labels1, mask) +\
                            sequence_cross_entropy_with_logits(label2_logits, labels2, mask) +\
                            sequence_cross_entropy_with_logits(label3_logits, labels3, mask)

        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        fdict = self.fscore.get_metric(reset)
        return {"accuracy1": self.accuracy1.get_metric(reset),
                "accuracy2": self.accuracy2.get_metric(reset),
                "accuracy3": self.accuracy3.get_metric(reset),
                "recall": fdict["recall"],
                "precision": fdict["precision"],
                "fscore": fdict["fscore"]}


