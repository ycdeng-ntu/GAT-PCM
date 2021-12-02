import os
import random


def argmin(data):
    best_val = data[0]
    best_idx = 0
    for idx, val in enumerate(data):
        if val < best_val:
            best_val = val
            best_idx = idx
    return best_idx


def argmax(data):
    best_val = data[0]
    best_idx = 0
    for idx, val in enumerate(data):
        if val > best_val:
            best_val = val
            best_idx = idx
    return best_idx


def proportional_selection(prob):
    rnd = random.random()
    total = 0
    idx = 0
    while total < rnd:
        total += prob[idx]
        idx += 1
    return idx - 1


def average(data):
    return sum(data) / len(data)


def load_data(path, dtype=float):
    if not os.path.exists(path):
        return None
    data = []
    f = open(path)
    for line in f:
        if line.strip() != '':
            data.append(dtype(line))
    return data
