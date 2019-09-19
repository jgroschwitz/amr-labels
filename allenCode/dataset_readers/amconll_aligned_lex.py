### author: Jonas Groschwitz
# reads an amconll file, yielding all input information and the lexical label as target output (replacing the lemma placeholder if there is one)

from typing import Iterator, List, Dict, Iterable
from allennlp.data import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from data_formatting.amconll_tools import parse_amconll, AMSentence
import re
from external.amr import AMR


# the following regex matches all sources
source_regex = re.compile("<[a-z0-9]+>")
preferred_secondary_nonlex = ["have-rel-role-91", "have-org-role-91", "rate-entity-91", "byline-91"]


def get_nonlex_labels(supertag: str) -> List[str]:
    """ gets all non-lexical labels from a supertag. Expects a supertag in the format amr--TYPE--type """


    # first get just the amr from the typed sypertag
    supertag = AMSentence.split_supertag(supertag)[0]
    if supertag == "_":
        return []
    # remove the root
    supertag = supertag.replace("<root>", "")
    # replace remaining sources, which are at unlabeled nodes, with a dummy label (so that the AMR parser doesn't get confused)
    # (we removed the root first because the root node is already labeled)
    supertag = re.sub(source_regex, " / --SOURCE--", supertag)
    # read the AMR with the method that comes with the Smatch script
    amr = AMR.parse_AMR_line(supertag)
    # now we can simply collect all labels ("node_values" in the external AMR code) that are not special --LEX-- or --SOURCE-- labels
    nonlex_labels = []
    for label in amr.node_values:
        if not label == "--LEX--" and not label == "--SOURCE--":
            nonlex_labels.append(label)
    return nonlex_labels


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
        fields["labels1"] = SequenceLabelField(labels, tokens, label_namespace="labels") # the 'labels' namespace is actually the default

        # get secondary labels from delexicalized supertag
        nonlex_labels_for_sentence = [get_nonlex_labels(supertag) for supertag in am_sentence.get_supertags()]
        labels2 = []
        labels3 = []
        for nonlex_labels_for_word in nonlex_labels_for_sentence:
            if len(nonlex_labels_for_word) == 0:
                labels2.append("_")
                labels3.append("_")
            elif len(nonlex_labels_for_word) == 1:
                labels2.append(nonlex_labels_for_word[0])
                labels3.append("_")
            else:
                # ignoring the case where the length is 3 or more
                if nonlex_labels_for_word[0] in preferred_secondary_nonlex:
                    nonlex_labels_for_word = [nonlex_labels_for_word[1], nonlex_labels_for_word[0]]
                labels2.append(nonlex_labels_for_word[0])
                labels3.append(nonlex_labels_for_word[1])

        fields["labels2"] = SequenceLabelField(labels2, tokens,
                                                   label_namespace="labels")  # the 'labels' namespace is actually the default
        fields["labels3"] = SequenceLabelField(labels3, tokens,
                                                   label_namespace="labels")  # the 'labels' namespace is actually the default



        # print([str(t) for t in tokens])
        # print([str(f) for f in fields["labels"]])
        return Instance(fields)


    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            am_sentences = parse_amconll(f)
            for sent in am_sentences:
                yield self.am_sent2instance(sent)


