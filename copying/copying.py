from typing import Dict, Iterable, Tuple
from data_formatting.amconll_tools import AMSentence

def get_lemma_to_label_stats(sents: Iterable[AMSentence]) -> Tuple[Dict[str, Dict[str,int]], Dict[str, int]]:
    ret = dict()
    for sent in sents:
        for word, lemma, label, rep in zip(sent.get_tokens(shadow_art_root=True), sent.get_lemmas(), sent.get_lexlabels(), sent.get_replacements()):
            label = label.replace("$LEMMA$", lemma).replace("$REPL$", rep).replace("$FORM$", word).lower()  # TODO: there is also $FORM$
            if rep == "_":
                lemma = lemma.lower()
            else:
                lemma = rep.lower()
            dict_here = ret.get(lemma, dict())
            if len(dict_here) == 0:
                ret[lemma] = dict_here
            dict_here[label] = dict_here.get(label, 0)+1

    lemma_counts = dict()
    for lemma , dict_here in ret.items():
        lemma_counts[lemma] = sum(dict_here.values())

    return ret, lemma_counts


def get_word_to_label_stats(sents: Iterable[AMSentence]) -> Tuple[Dict[str, Dict[str,int]], Dict[str, int]]:
    ret = dict()
    for sent in sents:
        for word, lemma, label, rep in zip(sent.get_tokens(shadow_art_root=True), sent.get_lemmas(), sent.get_lexlabels(), sent.get_replacements()):
            label = label.replace("$LEMMA$", lemma).replace("$REPL$", rep).replace("$FORM$", word).lower()
            if rep == "_":
                word = word.lower()
            else:
                word = rep.lower()
            dict_here = ret.get(word, dict())
            if len(dict_here) == 0:
                ret[word] = dict_here
            dict_here[label] = dict_here.get(label, 0)+1

    lemma_counts = dict()
    for word , dict_here in ret.items():
        lemma_counts[word] = sum(dict_here.values())

    return ret, lemma_counts



