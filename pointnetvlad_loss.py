import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.autograd import Variable
global memorybank
memorybank = []
criterion = torch.nn.CrossEntropyLoss(reduction="sum")


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[1]
    query_copies = query.repeat(1, int(num_pos), 1)
    diff = ((pos_vecs - query_copies) ** 2).sum(2)
    min_pos, _ = diff.min(1)
    max_pos, _ = diff.max(1)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


def triplet_loss_wrapper(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    return triplet_loss(q_vec, pos_vecs, neg_vecs, m1, use_min, lazy, ignore_zero_loss)


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    # PointNetVLAD official code use min_pos, but i think max_pos should be used
    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[1]
    batch = q_vec.shape[0]
    query_copies = q_vec.repeat(1, int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(1, int(num_neg))

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(2)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(1)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()

    other_neg_copies = other_neg.repeat(1, int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(2)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(1)[0]
    else:
        second_loss = second_loss.sum(1)

    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss.mean()

    total_loss = triplet_loss + second_loss
    return total_loss


def constrastive_loss(q_vec, pos_vecs, neg_vecs, other_neg):
    batch = q_vec.shape[0]
    num_pos = pos_vecs.shape[1]
    num_neg = neg_vecs.shape[1]
    # num_other_neg = other_neg.shape[1]

    query_copies_pos = q_vec.repeat(1, int(num_pos), 1)
    query_copies_neg = q_vec.repeat(1, int(num_neg), 1)
    # query_copies_other_neg = q_vec.repeat(1, int(num_other_neg), 1)

    pos_dot = torch.einsum('bmn, bmn->bm', query_copies_pos, pos_vecs)
    neg_dot = torch.einsum('bmn, bmn->bm', query_copies_neg, neg_vecs)
    # other_neg_dot = torch.einsum('bmn, bmn->bm', query_copies_other_neg, other_neg)

    # logits = torch.cat([pos_dot, neg_dot, other_neg_dot], dim=1)
    logits = torch.cat([pos_dot, neg_dot], dim=1)
    logits /= 0.07
    labels = torch.zeros(batch, dtype=torch.long, device=device)

    loss = criterion(logits, labels)

    return loss/batch


def constrastive_memorybank_loss(q_vec, pos_vecs, neg_vecs, other_neg):
    global memorybank
    # print(neg_vecs.size())

    if (len(memorybank) == 0):
        batch = q_vec.shape[0]
        num_pos = pos_vecs.shape[1]
        num_neg = neg_vecs.shape[1]
        # num_other_neg = other_neg.shape[1]

        query_copies_pos = q_vec.repeat(1, int(num_pos), 1)
        query_copies_neg = q_vec.repeat(1, int(num_neg), 1)
        # query_copies_other_neg = q_vec.repeat(1, int(num_other_neg), 1)

        pos_dot = torch.einsum('bmn, bmn->bm', query_copies_pos, pos_vecs)
        neg_dot = torch.einsum('bmn, bmn->bm', query_copies_neg, neg_vecs)
        # other_neg_dot = torch.einsum('bmn, bmn->bm', query_copies_other_neg, other_neg)

        # logits = torch.cat([pos_dot, neg_dot, other_neg_dot], dim=1)
        logits = torch.cat([pos_dot, neg_dot], dim=1)
        logits /= 0.5
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        loss = criterion(logits, labels)
    else:
        batch = q_vec.shape[0]
        num_pos = pos_vecs.shape[1]
        num_neg = neg_vecs.shape[1]
        # num_other_neg = other_neg.shape[1]
        num_memorybank = memorybank.shape[1]

        query_copies_pos = q_vec.repeat(1, int(num_pos), 1)
        query_copies_neg = q_vec.repeat(1, int(num_neg), 1)
        # query_copies_other_neg = q_vec.repeat(1, int(num_other_neg), 1)
        query_copies_neg_memorybank = q_vec.repeat(1, int(num_memorybank), 1)

        pos_dot = torch.einsum('bmn, bmn->bm', query_copies_pos, pos_vecs)
        neg_dot = torch.einsum('bmn, bmn->bm', query_copies_neg, neg_vecs)
        # other_neg_dot = torch.einsum('bmn, bmn->bm', query_copies_other_neg, other_neg)
        neg_memorybank_dot = torch.einsum('bmn, bmn->bm', query_copies_neg_memorybank, memorybank)

        # logits = torch.cat([pos_dot, neg_dot, other_neg_dot, neg_memorybank_dot], dim=1)
        logits = torch.cat([pos_dot, neg_dot, neg_memorybank_dot], dim=1)
        logits /= 0.5
        labels = torch.zeros(batch, dtype=torch.long, device=device)
        loss = criterion(logits, labels)

    # memorybank = Variable(torch.cat([pos_vecs, neg_vecs, other_neg], dim=1), requires_grad=False)
    memorybank = Variable(torch.cat([pos_vecs, neg_vecs], dim=1), requires_grad=False)

    return loss/batch


class constrastive_memory_loss(nn.Module):
    def __init__(self, batch: int, num_memory: int, dim: int):
        super(constrastive_memory_loss, self).__init__()
        self.ptr = 0
        self.T = 0.5
        # self.memory = Variable(torch.zeros(batch, num_memory, dim).cuda(), requires_grad=False)
        # self.memory = torch.zeros(batch, num_memory, dim).cuda()
        self.memory = torch.randn(batch, num_memory, dim).cuda()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        num_keys = keys.shape[1]
        num_memory = self.memory.shape[1]

        i = self.ptr % (num_memory // num_keys)
        self.memory[:, i*num_keys:(i+1)*num_keys, :] = keys
        self.ptr += 1

    def forward(self, q_vec, pos_vecs, neg_vecs, other_neg):
        if self.ptr < 1:
            batch = q_vec.shape[0]
            num_pos = pos_vecs.shape[1]
            num_neg = neg_vecs.shape[1]
            # num_other_neg = other_neg.shape[1]

            query_copies_pos = q_vec.repeat(1, int(num_pos), 1)
            query_copies_neg = q_vec.repeat(1, int(num_neg), 1)
            # query_copies_other_neg = q_vec.repeat(1, int(num_other_neg), 1)

            pos_dot = torch.einsum('bmn, bmn->bm', query_copies_pos, pos_vecs)
            neg_dot = torch.einsum('bmn, bmn->bm', query_copies_neg, neg_vecs)
            # other_neg_dot = torch.einsum('bmn, bmn->bm', query_copies_other_neg, other_neg)

            # logits = torch.cat([pos_dot, neg_dot, other_neg_dot], dim=1)
            logits = torch.cat([pos_dot, neg_dot], dim=1)
            logits /= self.T
            labels = torch.zeros(batch, dtype=torch.long, device=device)
            loss = criterion(logits, labels)
        else:
            batch = q_vec.shape[0]
            memory = self.memory.clone()

            num_pos = pos_vecs.shape[1]
            num_neg = neg_vecs.shape[1]
            # num_other_neg = other_neg.shape[1]
            num_memory = memory.shape[1]

            query_copies_pos = q_vec.repeat(1, int(num_pos), 1)
            query_copies_neg = q_vec.repeat(1, int(num_neg), 1)
            # query_copies_other_neg = q_vec.repeat(1, int(num_other_neg), 1)
            query_copies_neg_memory = q_vec.repeat(1, int(num_memory), 1)

            pos_dot = torch.einsum('bmn, bmn->bm', query_copies_pos, pos_vecs)
            neg_dot = torch.einsum('bmn, bmn->bm', query_copies_neg, neg_vecs)
            # other_neg_dot = torch.einsum('bmn, bmn->bm', query_copies_other_neg, other_neg)
            neg_memory_dot = torch.einsum('bmn, bmn->bm', query_copies_neg_memory, memory)

            # logits = torch.cat([pos_dot, neg_dot, other_neg_dot, neg_memory_dot], dim=1)
            logits = torch.cat([pos_dot, neg_dot, neg_memory_dot], dim=1)
            logits /= self.T
            labels = torch.zeros(batch, dtype=torch.long, device=device)
            loss = self.criterion(logits, labels)

        # self._dequeue_and_enqueue(torch.cat([pos_vecs, neg_vecs, other_neg], dim=1).clone())
        # self._dequeue_and_enqueue(torch.cat([pos_vecs, neg_vecs], dim=1).clone())
        self._dequeue_and_enqueue(torch.cat([pos_vecs, neg_vecs], dim=1))

        return loss/batch