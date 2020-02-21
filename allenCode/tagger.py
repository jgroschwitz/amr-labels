### based on https://github.com/allenai/allennlp/blob/master/tutorials/tagger/README.md#using-config-files

from typing import Iterator, List, Dict

import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.common.checks import ConfigurationError

import allenCode.losses as losses
from allenCode.f_metric import MultisetFScore
import allenCode.loss_mixer as loss_mixer


@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 loss) -> None:
        """ loss takes a list of 3 logit tensors, list of 3 gold tensors and a maks tensor to compute a loss."""
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
        self.loss = loss


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
            device = next(self.parameters()).device # slightly hacky, but https://github.com/pytorch/pytorch/issues/7460 recommends it
            output["loss"] = self.loss([label1_logits, label2_logits, label3_logits], [labels1, labels2, labels3], mask, device)

        return output


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        fdict = self.fscore.get_metric(reset)
        return {"accuracy1": self.accuracy1.get_metric(reset),
                "accuracy2": self.accuracy2.get_metric(reset),
                "accuracy3": self.accuracy3.get_metric(reset),
                "recall": fdict["recall"],
                "precision": fdict["precision"],
                "fscore": fdict["fscore"]}


    @classmethod
    def from_params(cls, params: Params, vocab: Vocabulary):
        word_embeddings = TextFieldEmbedder.from_params(params.pop("word_embeddings"), vocab=vocab)
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        #vocab = extras["vocab"]
        loss_str = params.pop("loss", "supervised")

        if loss_str == "supervised":
            loss = lambda logits, gold, mask, device: losses.supervised_loss(logits, gold, mask)
        elif loss_str == "reinforce":
            loss = lambda logits, gold, mask, device: losses.reinforce(logits, gold, mask, null_label_id=vocab.get_token_index(token="_", namespace='labels'), device=device)
        elif loss_str == "reinforce_with_baseline":
            loss = lambda logits, gold, mask, device: losses.reinforce_with_baseline(logits, gold, mask, null_label_id=vocab.get_token_index(token="_", namespace='labels'), device=device)
        elif loss_str == "restricted_reinforce":
            loss = lambda logits, gold, mask, device: losses.restricted_reinforce(logits, gold, mask, null_label_id=vocab.get_token_index(token="_", namespace='labels'), device=device)
        elif loss_str == "restricted_reinforce2":
            loss = lambda logits, gold, mask, device: losses.restricted_reinforce2(logits, gold, mask,
                                                                                  null_label_id=vocab.get_token_index(
                                                                                      token="_", namespace='labels'),
                                                                                  device=device)
        elif loss_str == "mix":
            loss = lambda logits, gold, mask, device: loss_mixer.mix(logits, gold, mask,
                                                                                  null_label_id=vocab.get_token_index(
                                                                                      token="_", namespace='labels'),
                                                                                  device=device)
        elif loss_str == "force_correct":
            loss = lambda logits, gold, mask, device: losses.force_correct(logits, gold, mask, null_label_id=vocab.get_token_index(token="_", namespace='labels'), device=device)
        else:
            raise ConfigurationError("Unrecognized loss "+loss_str)

        return LstmTagger(word_embeddings=word_embeddings,
                          encoder=encoder,
                          vocab=vocab,
                          loss = loss)


