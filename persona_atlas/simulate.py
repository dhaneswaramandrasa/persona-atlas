"""Simulate telco subscribers: daily token sequences plus usage features.

Each subscriber gets a persona, a home area, a personal twist on the persona's
content mix, and a few days of network events. An event becomes tokens the way a
telco would see it: the domain reached, a 3-hour time bucket and, sometimes, the
cell area. Some events are pure noise, so personas overlap like real people do.
"""
import numpy as np
import pandas as pd

from .personas import AREAS, CATEGORIES, DOMAINS, all_personas, hourly_curve

# Where each home area tends to travel for work/school during the day.
WORK_AREAS = [1.6, 2.4, 1.2, .8, .9, .9, .5, .6, .3]


def simulate(n: int = 20000, days: int = 5, events_per_day: int = 24, noise: float = .15, seed: int = 7):
    """Return (people DataFrame, list of token sentences, sentence -> person index)."""
    rng = np.random.default_rng(seed)
    personas = all_personas(seed)
    share = np.array([p.share for p in personas])
    pid = rng.choice(len(personas), size=n, p=share / share.sum())

    rows, sentences, owner = [], [], []
    for i in range(n):
        p = personas[pid[i]]
        home = rng.choice(len(AREAS), p=np.array(p.area) / sum(p.area))
        mix = rng.dirichlet(np.array(p.cats, float) * 3 + .3)       # this person's own content mix
        curve = hourly_curve(p)
        mobile = p.mobility_km > 8
        work = rng.choice(len(AREAS), p=np.array(WORK_AREAS) / sum(WORK_AREAS)) if mobile else home
        for _ in range(days):
            hours = np.sort(rng.choice(24, size=events_per_day, p=curve))
            sent = []
            for h in hours:
                cat = rng.integers(len(CATEGORIES)) if rng.random() < noise else rng.choice(len(CATEGORIES), p=mix)
                sent.append(DOMAINS[CATEGORIES[cat]][rng.integers(6)])
                sent.append(f"t{h // 3}")
                if rng.random() < .3:
                    away = mobile and 8 <= h <= 17
                    sent.append(f"cell_{AREAS[work if away else home]}")
            sentences.append(sent)
            owner.append(i)
        rows.append(dict(
            persona=p.id, persona_name=p.name, named=p.named, home=AREAS[home],
            gb_month=rng.lognormal(np.log(p.gb), .35),
            night_share=float(np.clip(rng.normal(p.night, .07), 0, 1)),
            mobility_km=rng.lognormal(np.log(p.mobility_km), .4),
        ))
    return pd.DataFrame(rows), sentences, np.array(owner)
