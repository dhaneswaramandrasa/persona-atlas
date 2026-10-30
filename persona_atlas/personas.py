"""Persona definitions for the simulated Jakarta telco population.

Twelve named personas carry hand-written behaviour profiles. Eighteen more are
generated at random so the catalogue has 30 groups, like the production one.
All numbers are invented.
"""
from dataclasses import dataclass, field

import numpy as np

CATEGORIES = [
    "social", "video", "games", "ecommerce", "ride_delivery",
    "ewallet_bank", "work_news", "learning", "gig_jobs", "parenting_health",
]

# Fake domains per content category. The network sees these as SNI / DNS names.
DOMAINS = {c: [f"{c}-{k:02d}.example" for k in range(6)] for c in CATEGORIES}

AREAS = [
    "jakarta_selatan", "jakarta_pusat", "jakarta_barat", "jakarta_timur", "jakarta_utara",
    "tangerang_bsd", "bekasi", "depok", "bogor",
]


@dataclass
class Persona:
    id: str
    name: str
    share: float                    # share of all subscribers
    hours: list                     # (peak hour, width, weight) bumps of daily activity
    cats: list                      # content weights, CATEGORIES order
    area: list                      # home-area weights, AREAS order
    gb: float                       # data per month
    night: float                    # share of traffic after 22:00
    mobility_km: float              # typical daily radius
    named: bool = True
    color: tuple = field(default=(190, 188, 180))


NAMED = [
    Persona("hs", "High school student", .07, [(7, 1, .4), (13, 2, .8), (20, 3, 1.6)],
            [9, 8, 9, 3, 1, 1, 1, 5, 0, 0], [1, .6, 1.2, 1.5, 1, 1, 1.4, 1, .8], 14, .32, 4, color=(232, 93, 117)),
    Persona("col", "College student", .065, [(10, 2, .7), (15, 2, .8), (22, 3, 1.5)],
            [9, 8, 5, 5, 5, 3, 2, 6, 1, 0], [1.6, .8, .7, .9, .4, 1, .7, 2.2, .8], 22, .41, 9, color=(240, 140, 60)),
    Persona("wc", "White collar", .10, [(7, 1, 1), (12, 1, .8), (18, 1.5, 1.2), (21, 2, .8)],
            [6, 4, 1, 5, 6, 7, 9, 2, 1, 1], [2.4, 2, 1, .6, .6, 1.4, .8, .6, .4], 26, .14, 14, color=(59, 63, 182)),
    Persona("bc", "Blue collar", .11, [(6, 1, .9), (12, 1, .7), (19, 2, 1.2)],
            [10, 5, 2, 2, 1, 3, 0, 0, 2, 1], [.3, .4, 1.2, 1.4, 1.4, 1.8, 2.2, .7, 1], 9, .20, 8, color=(42, 157, 143)),
    Persona("inf", "Informal worker", .12, [(6, 2, .9), (12, 3, 1), (19, 3, 1)],
            [8, 4, 1, 1, 7, 6, 0, 0, 8, 0], [1, 1, 1.2, 1.4, 1.2, 1, 1.2, 1, 1], 11, .27, 22, color=(140, 160, 40)),
    Persona("mom", "Mom with kids", .085, [(6, 1, .7), (10, 2, 1.1), (14, 2, .8), (20, 2, .9)],
            [8, 4, 1, 9, 6, 4, 1, 3, 0, 9], [1.2, .6, 1.1, 1.2, .8, 1.6, 1.4, 1.3, 1.1], 16, .12, 5, color=(176, 92, 196)),
    Persona("night", "Night-shift worker", .045, [(1, 2, 1.3), (5, 1, .7), (16, 2, .6)],
            [7, 8, 3, 2, 2, 3, 0, 0, 2, 0], [.8, 1.2, 1, 1, 1.4, 1.3, 1.4, .6, .5], 13, .62, 10, color=(90, 110, 140)),
    Persona("gamer", "Mobile gamer", .05, [(13, 2, .6), (21, 3, 1.6)],
            [3, 5, 10, 2, 1, 3, 0, 0, 0, 0], [1, .8, 1.2, 1.2, 1, 1, 1, .9, .7], 38, .48, 3, color=(214, 60, 160)),
    Persona("seller", "Online seller", .04, [(9, 3, 1), (15, 3, 1), (21, 2, .7)],
            [10, 3, 0, 10, 4, 6, 2, 0, 1, 0], [.8, 1.2, 1.6, 1, 1, 1.2, 1, .8, .6], 24, .25, 6, color=(205, 120, 30)),
    Persona("comm", "Bodetabek commuter", .07, [(5.5, 1, 1.2), (12, 1, .5), (18, 1.5, 1.3)],
            [6, 8, 1, 3, 5, 5, 6, 1, 0, 0], [.2, .2, .2, .2, .2, 2.4, 2.6, 2, 1.6], 18, .10, 32, color=(70, 140, 210)),
    Persona("affl", "Affluent family", .035, [(8, 2, .8), (13, 2, .7), (20, 2, 1.1)],
            [5, 7, 1, 7, 5, 8, 6, 2, 0, 3], [2.6, 1.4, .8, .5, .9, 2, .4, .4, .4], 34, .12, 12, color=(120, 80, 50)),
    Persona("ret", "Retiree", .03, [(5, 1.5, 1.1), (10, 2, .8), (16, 2, .6)],
            [10, 6, 0, 1, 1, 2, 3, 0, 0, 4], [1, 1, 1, 1.2, 1, 1, 1, 1, 1.2], 5, .06, 3, color=(150, 150, 110)),
]


def all_personas(seed: int = 7) -> list:
    """The 12 named personas plus 18 random ones that share the remaining population."""
    rng = np.random.default_rng(seed)
    rest = 1 - sum(p.share for p in NAMED)
    extra = []
    for k in range(18):
        bumps = [(float(rng.uniform(0, 24)), float(rng.uniform(1, 3)), float(rng.uniform(.5, 1.5)))
                 for _ in range(rng.integers(2, 4))]
        extra.append(Persona(
            f"p{k + 13:02d}", f"Persona {k + 13}", rest / 18, bumps,
            list(rng.gamma(1.2, 2, len(CATEGORIES)).round(2)), list(rng.gamma(2, .5, len(AREAS)).round(2)),
            float(rng.uniform(5, 35)), float(rng.uniform(.05, .5)), float(rng.uniform(3, 25)), named=False))
    return NAMED + extra


def hourly_curve(p: Persona) -> np.ndarray:
    """Activity weight for each hour of the day, from the persona's bumps."""
    h = np.arange(24)
    v = np.full(24, .08)
    for peak, width, w in p.hours:
        d = np.minimum(abs(h - peak), 24 - abs(h - peak))
        v += w * np.exp(-d ** 2 / (2 * width ** 2))
    return v / v.sum()
