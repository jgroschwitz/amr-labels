from copying.copying import get_lemma_to_label_stats, get_word_to_label_stats
from data_formatting.amconll_tools import parse_amconll
import numpy as np

if __name__ == "__main__":

    train_data = parse_amconll(open("data_formatting/amr17/train/train.amconll"))
    dev_data = parse_amconll(open("data_formatting/amr17/dev/gold-dev.amconll"))

    _, train_lemma_counts = get_word_to_label_stats(train_data)
    lemma2label_stats, lemma_counts = get_word_to_label_stats(dev_data)

    bucket_counts = dict()

    bucket_correct_lemma = dict()
    bucket_correct_lemma_v = dict()
    bucket_blanks = dict()

    lemma2bucket = dict()
    bucket2lemmas = dict()

    #print(lemma_counts)
    for lemma, count in lemma_counts.items():
        if lemma in train_lemma_counts.keys():
            train_count = train_lemma_counts[lemma]
            # convert count to log and back, rounding each step, to get bucket id among 1,3,7,20,...
            bucket_log = int(np.round(np.log(train_count)))
            bucket_id = int(np.round(np.exp(bucket_log)))
        else:
            bucket_id = 0


        # maps between lemmas and buckets, for convenience later
        lemma2bucket[lemma] = bucket_id
        other_lemmas = bucket2lemmas.get(bucket_id, None)
        if other_lemmas is None:
            other_lemmas = []
            bucket2lemmas[bucket_id] = other_lemmas
        other_lemmas.append(lemma)

        # update count for bucket_id in dev set
        bucket_counts[bucket_id] = bucket_counts.get(bucket_id, 0) + count



    # what we have now:
    # - a list of bucket ids based on training set counts: 0,1,3,7,20,... (not explicit, but implicit in the following maps)
    # - a map from lemma to bucket id
    # - a map from bucket id to all the lemmas
    # - total dev counts for each bucket id


    # the following gives me:
    # - all different correctness counts per bucket
    for lemma, label2count in lemma2label_stats.items():
        bucket_id = lemma2bucket[lemma]
        bucket_correct_lemma[bucket_id] = bucket_correct_lemma.get(bucket_id, 0) + label2count.get(lemma, 0)
        bucket_correct_lemma_v[bucket_id] = bucket_correct_lemma_v.get(bucket_id, 0) + label2count.get(lemma+"-01", 0)
        bucket_blanks[bucket_id] = bucket_blanks.get(bucket_id, 0) + label2count.get("_", 0)

    print(bucket_counts)
    print(bucket_blanks)
    buckets = list(bucket_counts.keys())
    buckets.sort()
    print(buckets)

    for bucket_id in buckets[-3:]:
        print(bucket_id)
        print([lemma+"/"+str(lemma_counts.get(lemma, 0)) for lemma in bucket2lemmas[bucket_id]])

    bucket_acc_lemma = [bucket_correct_lemma.get(i, 0)/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]
    bucket_acc_lemma_v = [bucket_correct_lemma_v.get(i, 0)/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]
    bucket_acc_lemma_either = [(bucket_correct_lemma.get(i, 0) + bucket_correct_lemma_v.get(i, 0))/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]

    print(bucket_acc_lemma)
    print(bucket_acc_lemma_v)
    print(bucket_acc_lemma_either)

    with open("copying/dev_copy_stats_words.csv", "w") as f:
        f.write("mincount,total,blanks,non_blanks,acc_lem, acc_lem_v, acc_lem_either\n")
        for i in range(len(buckets)):
            bucket_id = buckets[i]
            f.write(str(bucket_id)+","+str(bucket_counts[bucket_id])+","+str(bucket_blanks[bucket_id])
                    + "," + str(bucket_counts[bucket_id]-bucket_blanks[bucket_id])+","+str(bucket_acc_lemma[i])
                        + "," + str(bucket_acc_lemma_v[i])+","+str(bucket_acc_lemma_either[i])+"\n")


