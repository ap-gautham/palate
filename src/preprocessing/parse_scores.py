"""Parse originalScore strings into fractions in [0, 1].

Raw string and parsed value are kept separately so a parser improvement
never requires re-ingesting the CSVs.
"""
import re

import numpy as np
import pandas as pd

# Letter grades on a 13-step linear scale (A+ = 1.0 ... F = 0.0).
# The exact linear map is uncritical: per-critic z-scoring downstream is
# invariant to any per-critic linear transform of their scale.
_STEPS = 12.0
LETTER = {}
for i, base in enumerate(["A", "B", "C", "D"]):
    for j, mod in enumerate(["+", "", "-"]):
        LETTER[base + mod] = (12 - (i * 3 + j)) / _STEPS
LETTER["F+"] = 1 / _STEPS
LETTER["F"] = 0.0
LETTER["F-"] = 0.0
LETTER["E"] = 0.5 / _STEPS

_RE_FRACTION = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:/|(?:out\s+of))\s*(\d+(?:\.\d+)?)\s*$", re.I)
_RE_LETTER = re.compile(r"^\s*([A-F][+-]?)\s*$", re.I)
_RE_PERCENT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*%\s*$")
_RE_STARS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:\*|stars?)\s*(?:\(out of (\d+)\))?\s*$", re.I)
_RE_NUM_OF_5 = re.compile(r"^\s*(\d(?:\.\d+)?)\s*$")  # bare number — ambiguous, rejected


def parse_one(s) -> float:
    """Return a fraction in [0, 1], or NaN when the string is unparseable."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.strip().strip("'\"")

    m = _RE_FRACTION.match(s)
    if m:
        num = float(m.group(1).replace(",", "."))
        den = float(m.group(2))
        if den <= 0 or den > 100 or num < 0:
            return np.nan
        if num > den:  # e.g. "5/4" typo — untrustworthy
            return np.nan
        return num / den

    m = _RE_LETTER.match(s)
    if m:
        return LETTER.get(m.group(1).upper(), np.nan)

    m = _RE_PERCENT.match(s)
    if m:
        v = float(m.group(1))
        return v / 100.0 if v <= 100 else np.nan

    m = _RE_STARS.match(s)
    if m:
        v = float(m.group(1))
        den = float(m.group(2)) if m.group(2) else 5.0
        if v <= den:
            return v / den
        return np.nan

    return np.nan


def parse_series(scores: pd.Series) -> pd.Series:
    """Vectorized-ish parse: parse each unique string once, then map back."""
    uniq = scores.dropna().unique()
    lookup = {u: parse_one(u) for u in uniq}
    return scores.map(lookup).astype("float64")


def standardize_to_levels(frac: pd.Series, levels: int, rng) -> pd.Series:
    """Quantize a fraction in [0, 1] onto the integer scale {0, .., levels}.

    Puts every heterogeneous source scale (4-star, 5-point, letter grade,
    percentage) onto one common ordinal scale. Rounds to the nearest integer;
    exact half-way values (e.g. 3.5/5 -> 3.5 on a 0-5 scale) are broken
    randomly up or down with equal probability, so the quantization adds no
    systematic upward or downward bias.
    """
    q = frac.to_numpy(dtype="float64") * levels
    floor = np.floor(q)
    rem = q - floor
    out = floor.copy()
    out[rem > 0.5 + 1e-9] += 1
    tie = np.isclose(rem, 0.5)
    coin = rng.random(len(q)) < 0.5          # True -> round the tie up
    out[tie & coin] += 1
    out[~np.isfinite(q)] = np.nan
    return pd.Series(out, index=frac.index)
