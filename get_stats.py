from copying.copying import get_lemma_to_label_stats
from data_formatting.amconll_tools import parse_amconll
import numpy as np

if __name__ == "__main__":

    train_data = parse_amconll(open("data_formatting/amr17/train/train.amconll"))

    lemma2label_stats, lemma_counts = get_lemma_to_label_stats(train_data)

    bucket_counts = dict()

    bucket_correct_lemma = dict()
    bucket_correct_lemma_v = dict()
    bucket_blanks = dict()

    #print(lemma_counts)
    for lemma, count in lemma_counts.items():
        bucket_id = int(np.round(np.log(count)))
        if bucket_id>=7:
            print(lemma +": "+str(count))
        bucket_counts[bucket_id] = bucket_counts.get(bucket_id, 0) + count

    for lemma, label2count in lemma2label_stats.items():
        bucket_id = int(np.round(np.log(lemma_counts[lemma])))
        bucket_correct_lemma[bucket_id] = bucket_correct_lemma.get(bucket_id, 0) + label2count.get(lemma, 0)
        bucket_correct_lemma_v[bucket_id] = bucket_correct_lemma_v.get(bucket_id, 0) + label2count.get(lemma+"-01", 0)
        bucket_blanks[bucket_id] = bucket_blanks.get(bucket_id, 0) + label2count.get("_", 0)

    print(bucket_counts)
    print(bucket_blanks)
    buckets = list(bucket_counts.keys())
    buckets.sort()
    bucket_acc_lemma = [bucket_correct_lemma.get(i, 0)/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]
    bucket_acc_lemma_v = [bucket_correct_lemma_v.get(i, 0)/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]
    bucket_acc_lemma_either = [(bucket_correct_lemma.get(i, 0) + bucket_correct_lemma_v.get(i, 0))/(bucket_counts[i]-bucket_blanks.get(i,0)) for i in buckets]

    print(bucket_acc_lemma)
    print(bucket_acc_lemma_v)
    print(bucket_acc_lemma_either)

    with open("copying/train_copy_stats.csv", "w") as f:
        f.write("mincount,total,blanks,non_blanks,acc_lem, acc_lem_v, acc_lem_either\n")
        for i in buckets:
            f.write(str(int(np.round(np.exp(i))))+","+str(bucket_counts[i])+","+str(bucket_blanks[i])
                    + "," + str(bucket_counts[i]-bucket_blanks[i])+","+str(bucket_acc_lemma[i])
                        + "," + str(bucket_acc_lemma_v[i])+","+str(bucket_acc_lemma_either[i])+"\n")


