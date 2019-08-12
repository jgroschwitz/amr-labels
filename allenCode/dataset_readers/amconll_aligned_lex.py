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
    reads an amconll file, yielding all input information and the lexical label as target output (replacing the lemma placeholder if there is one)
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}


    def am_sent2instance(self, am_sentence: AMSentence) -> Instance:
        fields: Dict[str, Field] = {}
        raw_words = am_sentence.get_tokens(shadow_art_root=True)
        reps = am_sentence.get_replacements()
        words = []
        for word, rep in zip(raw_words, reps):
            if rep in ["_name_", "_number_", "_date_"]:
                words.append(rep) # currently just use the lemma for these to make the task easier
            else:
                words.append(word.lower()) # TODO possibly remove lowercasing again later, if we do something smarter than 1-hot later (bert, character, ...)

        tokens = TextField([Token(w) for w in words], self.token_indexers)
        fields["sentence"] = tokens
        # fields["pos_tags"] = SequenceLabelField(am_sentence.get_pos(), tokens, label_namespace="pos")
        # fields["ner_tags"] = SequenceLabelField(am_sentence.get_ner(), tokens, label_namespace="ner_labels")
        # fields["lemmas"] = SequenceLabelField(am_sentence.get_lemmas(), tokens, label_namespace="lemmas")
        labels = am_sentence.get_lexlabels()
        lemmas = am_sentence.get_lemmas()
        labels = [label.replace("$LEMMA$", lemma).replace("$REPL$", rep) for label, lemma, rep in zip(labels, lemmas, reps)] #  TODO: there is also $FORM$
        fields["labels"] = SequenceLabelField(labels, tokens) #,label_namespace="lex_labels"

        # print([str(t) for t in tokens])
        # print([str(f) for f in fields["labels"]])
        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            am_sentences = parse_amconll(f)
            for sent in am_sentences:
                yield self.am_sent2instance(sent)