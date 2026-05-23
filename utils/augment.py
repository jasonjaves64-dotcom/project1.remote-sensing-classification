import numpy as np
import torch

def randomFlip(x, y, h_prob=0.5, v_prob=0.5):
    if np.random.rand() < h_prob:
        x = np.flip(x, axis=-1).copy()
        y = np.flip(y, axis=-1).copy()
    if np.random.rand() < v_prob:
        x = np.flip(x, axis=-2).copy()
        y = np.flip(y, axis=-2).copy()
    return x, y

def randomTemporalDropout(x, max_drop=2):
    T = x.shape[0]
    n_drop = np.random.randint(0, max_drop + 1)
    if n_drop > 0:
        drop_idx = np.random.choice(T, n_drop, replace=False)
        x = x.copy()
        x[drop_idx] = 0.0
    return x

def randomNoise(x, sigma=0.01):
    noise = np.random.randn(*x.shape).astype(np.float32) * sigma
    return np.clip(x + noise, 0, 1)

def temporalShuffle(x, shuffle_ratio=0.1):
    T = x.shape[0]
    n_shuffle = int(T * shuffle_ratio)
    if n_shuffle > 1:
        indices = np.random.choice(T, n_shuffle, replace=False)
        shuffled = indices.copy()
        np.random.shuffle(shuffled)
        x = x.copy()
        x[indices] = x[shuffled]
    return x

class TemporalAugment:
    def __init__(self, p_temporal_dropout=0.3, max_drop=2):
        self.p_temporal_dropout = p_temporal_dropout
        self.max_drop = max_drop

    def __call__(self, x):
        if np.random.rand() < self.p_temporal_dropout:
            x = randomTemporalDropout(x, self.max_drop)
        return x

class SpatialAugment:
    def __init__(self, p_flip=0.5, p_noise=0.2, sigma=0.01):
        self.p_flip = p_flip
        self.p_noise = p_noise
        self.sigma = sigma

    def __call__(self, x, y):
        x, y = randomFlip(x, y, h_prob=self.p_flip, v_prob=self.p_flip)
        if np.random.rand() < self.p_noise:
            x = randomNoise(x, self.sigma)
        return x, y

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y=None):
        for t in self.transforms:
            if y is not None:
                x, y = t(x, y)
            else:
                x = t(x)
        return x if y is None else (x, y)