from typing import Dict, Iterable, Tuple
from data_formatting.amconll_tools import AMSentence

def get_lemma_to_label_stats(sents: Iterable[AMSentence]) -> Tuple[Dict[str, Dict[str,int]], Dict[str, int]]:
    ret = dict()
    for sent in sents:
        for lemma, label in zip(sent.get_lemmas(), sent.get_lexlabels()):
            label = label.replace("$LEMMA$", lemma).replace("$REPL$", lemma).lower()  # TODO: ask Matthias about diff between $LEMMA$ and #REPL$ TODO: there is also $FORM$
            lemma = lemma.lower()
            dict_here = ret.get(lemma, dict())
            if len(dict_here) == 0:
                ret[lemma] = dict_here
            dict_here[label] = dict_here.get(label, 0)+1

    lemma_counts = dict()
    for lemma , dict_here in ret.items():
        lemma_counts[lemma] = sum(dict_here.values())

    return ret, lemma_counts



