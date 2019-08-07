### author: Jonas Groschwitz
# reads an amconll file, yielding all input information and the lexical label as target output (replacing the lemma placeholder if there is one)

from typing import Iterator, List, Dict, Iterable
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from data_formatting.amconll_tools import parse_amconll, AMSentence


@DatasetReader.register('amconll-aligned-lex')
class AMAlignedLexReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    def am_sent2instance(self, am_sentence: AMSentence) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = TextField([Token(w) for w in am_sentence.get_tokens(shadow_art_root=True)], self.token_indexers)
        fields["sentence"] = tokens
        # fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        # fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner_labels")
        # fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")
        fields["labels"] = SequenceLabelField(am_sentence.get_lexlabels(), tokens) #,label_namespace="lex_labels"

        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            am_sentences = parse_amconll(f)
            for sent in am_sentences:
                yield self.am_sent2instance(sent)