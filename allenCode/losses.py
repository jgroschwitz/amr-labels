from allennlp.nn.util import sequence_cross_entropy_with_logits
from typing import List, Dict
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np

from allenCode.f_metric import MultisetFScore

def supervised_loss(logits: List[Tensor], gold: List[Tensor], mask):
    label1_logits, label2_logits, label3_logits = logits
    labels1, labels2, labels3 = gold
    return sequence_cross_entropy_with_logits(label1_logits, labels1, mask) +\
                            sequence_cross_entropy_with_logits(label2_logits, labels2, mask) +\
                            sequence_cross_entropy_with_logits(label3_logits, labels3, mask)


def reinforce(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
    # logits are of shape batch_size x sent_length x num_categories
    label1_logits, label2_logits, label3_logits = logits
    np_logits = [F.softmax(l, dim=2).detach().cpu().numpy() for l in logits]
    # golds and mask are of shape batch_size x sent_length
    gold1, gold2, gold3 = gold
    batch_size, sent_length, num_cats = label1_logits.size()
    rewards = []
    samples = [[],[],[]]
    for k in range(batch_size):
        for l in range(3):
            samples[l].append([])
        gold_counts = dict()
        sample_counts = dict()
        for i in range(sent_length):
            if mask[k][i].item() > 0:
                sample1 = sample(np_logits[0], k, i)
                if sample1 == null_label_id:
                    sample2 = null_label_id
                else:
                    sample2 = sample(np_logits[1], k, i)
                if sample2 == null_label_id:
                    sample3 = null_label_id
                else:
                    sample3 = sample(np_logits[2], k, i)
                samples[0][k].append(sample1)
                samples[1][k].append(sample2)
                samples[2][k].append(sample3)

                add_count(sample_counts, sample1, null_label_id)
                add_count(sample_counts, sample2, null_label_id)
                add_count(sample_counts, sample3, null_label_id)
                add_count(gold_counts, gold1[k][i].item(), null_label_id)
                add_count(gold_counts, gold2[k][i].item(), null_label_id)
                add_count(gold_counts, gold3[k][i].item(), null_label_id)
            else:
                samples[0][k].append(0)
                samples[1][k].append(0)
                samples[2][k].append(0)
        rewards.append(fscore(sample_counts, gold_counts))
    rewards = torch.tensor(rewards, requires_grad=False, device=device)
    mask_with_reward = torch.mul(mask.float(), rewards.view([batch_size, 1]))
    return sequence_cross_entropy_with_logits(label1_logits, torch.tensor(samples[0], dtype=torch.long, requires_grad=False, device=device), mask_with_reward) +\
                            sequence_cross_entropy_with_logits(label2_logits, torch.tensor(samples[1], dtype=torch.long, requires_grad=False, device=device), mask_with_reward) +\
                            sequence_cross_entropy_with_logits(label3_logits, torch.tensor(samples[2], dtype=torch.long, requires_grad=False, device=device), mask_with_reward)





def reinforce_with_baseline(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
    # logits are of shape batch_size x sent_length x num_categories
    label1_logits, label2_logits, label3_logits = logits
    np_logits = [F.softmax(l/5.0, dim=2).detach().cpu().numpy() for l in logits]
    # golds and mask are of shape batch_size x sent_length
    gold1, gold2, gold3 = gold
    batch_size, sent_length, num_cats = label1_logits.size()
    rewards = []
    samples = [[], [], []]
    for k in range(batch_size):
        for l in range(3):
            samples[l].append([])
        gold_counts = dict()
        sample_counts = dict()
        best_counts = dict()
        for i in range(sent_length):
            if mask[k][i].item() > 0:
                sample1 = sample(np_logits[0], k, i)
                if sample1 == null_label_id:
                    sample2 = null_label_id
                else:
                    sample2 = sample(np_logits[1], k, i)
                if sample2 == null_label_id:
                    sample3 = null_label_id
                else:
                    sample3 = sample(np_logits[2], k, i)
                samples[0][k].append(sample1)
                samples[1][k].append(sample2)
                samples[2][k].append(sample3)

                best1 = best(np_logits[0], k, i)
                if best1 == null_label_id:
                    best2 = null_label_id
                else:
                    best2 = best(np_logits[1], k, i)
                if best2 == null_label_id:
                    best3 = null_label_id
                else:
                    best3 = best(np_logits[2], k, i)

                add_count(sample_counts, sample1, null_label_id)
                add_count(sample_counts, sample2, null_label_id)
                add_count(sample_counts, sample3, null_label_id)
                add_count(best_counts, best1, null_label_id)
                add_count(best_counts, best2, null_label_id)
                add_count(best_counts, best3, null_label_id)
                add_count(gold_counts, gold1[k][i].item(), null_label_id)
                add_count(gold_counts, gold2[k][i].item(), null_label_id)
                add_count(gold_counts, gold3[k][i].item(), null_label_id)
            else:
                samples[0][k].append(0)
                samples[1][k].append(0)
                samples[2][k].append(0)
        rewards.append(fscore(sample_counts, gold_counts) - fscore(best_counts, gold_counts)) # subtract baseline here
    rewards = torch.tensor(rewards, requires_grad=False, device=device)
    mask_with_reward = torch.mul(mask.float(), rewards.view([batch_size, 1]))
    return sequence_cross_entropy_with_logits(label1_logits,
                                              torch.tensor(samples[0], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward) + \
           sequence_cross_entropy_with_logits(label2_logits,
                                              torch.tensor(samples[1], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward) + \
           sequence_cross_entropy_with_logits(label3_logits,
                                              torch.tensor(samples[2], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward)



def restricted_reinforce(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
    # logits are of shape batch_size x sent_length x num_categories
    label1_logits, label2_logits, label3_logits = logits
    batch_size, sent_length, num_cats = label1_logits.size()
    np_logits = [F.softmax(l, dim=2).detach().cpu().numpy() for l in logits]
    # golds and mask are of shape batch_size x sent_length
    gold1, gold2, gold3 = gold
    # collect gold counts a bit earlier here
    gold_counts = []
    for k in range(batch_size):
        gold_counts.append(dict())
        for i in range(sent_length):
            add_count(gold_counts[k], gold1[k][i].item(), null_label_id)
            add_count(gold_counts[k], gold2[k][i].item(), null_label_id)
            add_count(gold_counts[k], gold3[k][i].item(), null_label_id)


    rewards = []
    samples = [[], [], []]
    for k in range(batch_size):
        # set up the vector of valid predictions (0.0 except for labels in gold set plus the null prediction "_",
        # where it is 1.0) we use this vector to multiply our probabilites later, so we sample only from valid predictions
        valid_predictions = np.zeros(num_cats)
        for j in gold_counts[k].keys():
            valid_predictions[j] = 1.0
        valid_predictions[null_label_id] = 1.0

        # set up our storage for the samples we make
        for l in range(3):
            samples[l].append([])
        sample_counts = dict()

        for i in range(sent_length):
            # we only need to do all this stuff if the word is not just buffer
            if mask[k][i].item() > 0:

                # set all invalid predictions to 0, leave rest unchanged
                for l in range(3):
                    np_logits[l][k][i] = np.multiply(np_logits[l][k][i], valid_predictions)
                    # renormalize
                    np_sum = np.sum(np_logits[l][k][i])
                    np_logits[l][k][i] = np_logits[l][k][i] / np_sum

                # now we actually sample. Predictions for secondary/tertiary labels may only be non-null if the
                # previous prediction (primary/secondary respectively) was not null.
                sample1 = sample(np_logits[0], k, i)
                if sample1 == null_label_id:
                    sample2 = null_label_id
                else:
                    sample2 = sample(np_logits[1], k, i)
                if sample2 == null_label_id:
                    sample3 = null_label_id
                else:
                    sample3 = sample(np_logits[2], k, i)

                # store our samples
                samples[0][k].append(sample1)
                samples[1][k].append(sample2)
                samples[2][k].append(sample3)
                add_count(sample_counts, sample1, null_label_id)
                add_count(sample_counts, sample2, null_label_id)
                add_count(sample_counts, sample3, null_label_id)

            else:
                # just buffer the samples to full length
                samples[0][k].append(0)
                samples[1][k].append(0)
                samples[2][k].append(0)
        rewards.append(fscore(sample_counts, gold_counts[k]))

    rewards = torch.tensor(rewards, requires_grad=False, device=device)
    mask_with_reward = torch.mul(mask.float(), rewards.view([batch_size, 1]))
    return sequence_cross_entropy_with_logits(label1_logits,
                                              torch.tensor(samples[0], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward) + \
           sequence_cross_entropy_with_logits(label2_logits,
                                              torch.tensor(samples[1], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward) + \
           sequence_cross_entropy_with_logits(label3_logits,
                                              torch.tensor(samples[2], dtype=torch.long, requires_grad=False, device=device),
                                              mask_with_reward)



def force_correct(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
    # logits are of shape batch_size x sent_length x num_categories
    label1_logits, label2_logits, label3_logits = logits
    batch_size, sent_length, num_cats = label1_logits.size()
    np_logits = [F.softmax(l, dim=2).detach().cpu().numpy() for l in logits]
    # golds and mask are of shape batch_size x sent_length
    gold1, gold2, gold3 = gold
    # collect gold counts a bit earlier here
    gold_counts = []
    for k in range(batch_size):
        gold_counts.append(dict())
        for i in range(sent_length):
            add_count(gold_counts[k], gold1[k][i].item(), null_label_id)
            add_count(gold_counts[k], gold2[k][i].item(), null_label_id)
            add_count(gold_counts[k], gold3[k][i].item(), null_label_id)


    samples = [[], [], []]
    for k in range(batch_size):
        for l in range(3):
            samples[l].append([])
        sample_counts = dict()
        for i in range(sent_length):
            if mask[k][i].item() > 0:
                valid_predictions = np.zeros(num_cats)
                for j in gold_counts[k].keys():
                    if gold_counts[k].get(j, 0) - sample_counts.get(j, 0) > 0:
                        valid_predictions[j] = 1.0
                gold_remaining = sum(gold_counts[k].values()) - sum(sample_counts.values())
                predictions_remaining = 3 * (sent_length-(i+1)) # predictions remaining after this word
                if gold_remaining <= 0 or predictions_remaining - gold_remaining > 0:
                    valid_predictions[null_label_id] = 1.0
                # set all invalid predictions to 0, leave rest unchanged
                for l in range(3):
                    np_logits[l][k][i] = np.multiply(np_logits[l][k][i], valid_predictions)
                    # renormalize
                    np_sum = np.sum(np_logits[l][k][i])
                    np_logits[l][k][i] = np_logits[l][k][i] / np_sum

                # now we sample
                sample1 = sample(np_logits[0], k, i)
                if sample1 == null_label_id:
                    sample2 = null_label_id
                else:
                    sample2 = sample(np_logits[1], k, i)
                if sample2 == null_label_id:
                    sample3 = null_label_id
                else:
                    sample3 = sample(np_logits[2], k, i)

                # store samples
                samples[0][k].append(sample1)
                samples[1][k].append(sample2)
                samples[2][k].append(sample3)
                add_count(sample_counts, sample1, null_label_id)
                add_count(sample_counts, sample2, null_label_id)
                add_count(sample_counts, sample3, null_label_id)
            else:
                samples[0][k].append(0)
                samples[1][k].append(0)
                samples[2][k].append(0)

    return sequence_cross_entropy_with_logits(label1_logits,
                                              torch.tensor(samples[0], dtype=torch.long, requires_grad=False, device=device),
                                              mask) + \
           sequence_cross_entropy_with_logits(label2_logits,
                                              torch.tensor(samples[1], dtype=torch.long, requires_grad=False, device=device),
                                              mask) + \
           sequence_cross_entropy_with_logits(label3_logits,
                                              torch.tensor(samples[2], dtype=torch.long, requires_grad=False, device=device),
                                              mask)



# def label_mse(logits: List[Tensor], gold: List[Tensor], mask, null_label_id, device):
#     # logits are of shape batch_size x sent_length x num_categories
#     label1_logits, label2_logits, label3_logits = logits
#     batch_size, sent_length, num_cats = label1_logits.size()
#     np_logits = [F.softmax(l, dim=2).detach().cpu().numpy() for l in logits]
#     # golds and mask are of shape batch_size x sent_length
#     gold1, gold2, gold3 = gold
#     # collect gold counts a bit earlier here
#     gold_counts = []
#     for k in range(batch_size):
#         gold_counts.append(dict())
#         for i in range(sent_length):
#             add_count(gold_counts[k], gold1[k][i].item(), null_label_id)
#             add_count(gold_counts[k], gold2[k][i].item(), null_label_id)
#             add_count(gold_counts[k], gold3[k][i].item(), null_label_id)
#
#     for k in range(batch_size):






def add_count(counter: Dict[int, int], key: int, null_label_id: int):
    if key != null_label_id:
        counter[key] = counter.get(key, 0) + 1



def fscore(predicted: Dict[int, int], gold: Dict[int, int]):
    correct = 0
    for label in gold.keys():
        correct += min(gold.get(label, 0), predicted.get(label, 0))
    if sum(gold.values()) > 0:
        recall = float(correct) / float(sum(gold.values()))
    else:
        recall = 0.0
    if sum(predicted.values()) > 0:
        precision = float(correct) / float(sum(predicted.values()))
    else:
        precision = 0.0
    if recall > 1e-12 or precision > 1e-12:
        return 2*recall*precision / (recall + precision)
    else:
        return 0.0



def sample(logits: np.ndarray, batch_index: int, sent_index: int) -> int:
    probs = logits[batch_index][sent_index]
    if np.isnan(probs).any():
        return np.random.choice(len(probs))
    return np.random.choice(len(probs), p=probs)


def best(logits: np.ndarray, batch_index: int, sent_index: int) -> int:
    probs = logits[batch_index][sent_index]
    return np.argmax(probs)



