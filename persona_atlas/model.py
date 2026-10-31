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


def export_web(path, pos, persona_idx, area_idx, meta: dict, extra: dict = None):
    """Write a compact JSON the web demo reads: int coords ×10, persona index, home-area index."""
    data = dict(meta=meta,
                xyz=np.round(pos * 10).astype(int).ravel().tolist(),
                persona=persona_idx.tolist(), area=area_idx.tolist(), **(extra or {}))
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def supervised_persona(X, truth_mask, cluster_mask, extra=None, label_frac=.03, seed=7):
    """Score one persona with a supervised model instead of clusters.

    Some personas have labels: a small survey, or a strong proxy such as a corporate-paid plan for
    white collar. Train on the labelled few, score everyone, and compare with the cluster-based label.
    The threshold keeps the predicted share equal to the labelled sample's share, so sizes stay unbiased.
    """
    from sklearn.linear_model import LogisticRegression
    truth_mask, cluster_mask = np.asarray(truth_mask, bool), np.asarray(cluster_mask, bool)   # pandas masks index differently
    rng = np.random.default_rng(seed + 1000)   # independent of the simulation's stream (seed 7 drew the personas)
    F = X if extra is None else np.hstack([X, extra])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    lab = rng.random(len(X)) < label_frac
    clf = LogisticRegression(C=.5, max_iter=2000).fit(F[lab], truth_mask[lab])   # few labels: keep the model simple
    prob = clf.predict_proba(F)[:, 1]
    k = int(round(truth_mask[lab].mean() * len(X)))                             # predicted share = labelled share
    pred = np.zeros(len(X), bool)
    pred[np.argsort(-prob)[:k]] = True
    t = ~lab

    def pr(p):
        hit = (p & truth_mask)[t].sum()
        return dict(precision=hit / max(1, p[t].sum()), recall=hit / truth_mask[t].sum())
    return pred, prob, dict(labelled=int(lab.sum()), supervised=pr(pred), clustering=pr(cluster_mask))


def subscriber_profiles(sentences, owner, n, categories, areas):
    """Per subscriber, from their own token sentences: content mix, work area, hourly activity.

    Returns int arrays the web demo uses to draw each dot as a person: top 3 content categories with
    shares (%), the most-used daytime cell outside home, and activity per 3-hour bucket (%).
    """
    cat_idx = {c: i for i, c in enumerate(categories)}
    area_idx = {a: i for i, a in enumerate(areas)}
    cats = np.zeros((n, len(categories)))
    cells = np.zeros((n, len(areas)))
    hours = np.zeros((n, 8))
    for sent, i in zip(sentences, owner):
        last_t = 0
        for tok in sent:
            if tok.endswith(".example"):
                cats[i, cat_idx[tok.rsplit("-", 1)[0]]] += 1
            elif tok.startswith("cell_"):
                if 3 <= last_t <= 5:                       # 09:00-17:59: where they spend the day
                    cells[i, area_idx[tok[5:]]] += 1
            elif tok.startswith("t"):
                last_t = int(tok[1:])
                hours[i, last_t] += 1
    share = cats / cats.sum(1, keepdims=True)
    top = np.argsort(-share, 1)[:, :3]
    top_share = np.round(np.take_along_axis(share, top, 1) * 100).astype(int)
    hours = np.round(hours / hours.sum(1, keepdims=True) * 100).astype(int)
    return top, top_share, cells, hours
