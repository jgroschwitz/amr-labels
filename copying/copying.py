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
            add_nested_count(ret, lemma, label)

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
            add_nested_count(ret, word, label)

    lemma_counts = dict()
    for word , dict_here in ret.items():
        lemma_counts[word] = sum(dict_here.values())

    return ret, lemma_counts


def add_nested_count(nested_dict: Dict[str, Dict[str,int]], key, value):
    dict_here = nested_dict.get(key, dict())
    if len(dict_here) == 0:
        nested_dict[key] = dict_here
    dict_here[value] = dict_here.get(value, 0) + 1


