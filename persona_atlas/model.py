"""Embed, cluster, look-alike and export.

1. Word2Vec on token sentences (domains, time buckets, cell areas).
2. A subscriber vector = mean of their token vectors + scaled usage features.
3. K-means into personas.
4. Look-alike: score everyone by cosine similarity to a small seed list.
5. A 3D layout for the web demo.
"""
import json

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

USAGE = ["gb_month", "night_share", "mobility_km"]


def train_word2vec(sentences, dim: int = 48, seed: int = 7) -> Word2Vec:
    return Word2Vec(sentences, vector_size=dim, window=6, min_count=1, sg=1, negative=8,
                    epochs=5, workers=1, seed=seed)


def subscriber_vectors(w2v, sentences, owner, people, usage_weight: float = .35) -> np.ndarray:
    n = len(people)
    dim = w2v.vector_size
    acc = np.zeros((n, dim))
    cnt = np.zeros(n)
    for sent, i in zip(sentences, owner):
        acc[i] += w2v.wv[sent].sum(0)
        cnt[i] += len(sent)
    behaviour = normalize(acc / cnt[:, None])
    usage = StandardScaler().fit_transform(np.log1p(people[USAGE].values)) * usage_weight / np.sqrt(len(USAGE))
    return normalize(np.hstack([behaviour, usage]))


def cluster(X, k: int = 30, seed: int = 7):
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    return km.labels_, km.cluster_centers_


def lookalike_curve(X, truth_mask, seed_frac: float = .02, sizes=None, seed: int = 7):
    """Start from a small known seed list (e.g. a client survey), rank everyone else by similarity.

    Returns a list of dicts: audience size, precision and recall against the true persona.
    """
    rng = np.random.default_rng(seed)
    members = np.flatnonzero(truth_mask)
    seeds = rng.choice(members, size=max(5, int(len(members) * seed_frac)), replace=False)
    centre = normalize(X[seeds].mean(0, keepdims=True))[0]
    score = X @ centre
    score[seeds] = -np.inf                          # rank only the people we don't know yet
    order = np.argsort(-score)
    total = truth_mask.sum() - len(seeds)
    sizes = sizes or [int(total * f) for f in (.25, .5, .75, 1, 1.5, 2, 3)]
    out = []
    for s in sizes:
        hit = truth_mask[order[:s]].sum()
        out.append(dict(audience=s, precision=hit / s, recall=hit / total))
    return out, score


def layout_3d(X, labels, centers, spread: float = 7.0, radius: float = 55.0, seed: int = 7):
    """Place cluster centres with MDS (similar personas sit close), then each point by local PCA."""
    c3 = MDS(3, random_state=seed, n_init=4, init="random").fit_transform(centers)
    c3 = c3 / np.abs(c3).max() * radius
    pos = np.zeros((len(X), 3))
    for k in range(len(centers)):
        idx = np.flatnonzero(labels == k)
        local = PCA(3, random_state=seed).fit_transform(X[idx] - centers[k])
        local = np.clip(local / (local.std(0) + 1e-9), -2.5, 2.5) * spread / 2   # round blobs, no outlier streaks
        pos[idx] = c3[k] + local
    return pos


def export_web(path, pos, persona_idx, area_idx, meta: dict):
    """Write a compact JSON the web demo reads: int coords ×10, persona index, home-area index."""
    data = dict(meta=meta,
                xyz=np.round(pos * 10).astype(int).ravel().tolist(),
                persona=persona_idx.tolist(), area=area_idx.tolist())
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
