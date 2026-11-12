"""Looking for a classification law in the precursor: the simplest product of powers of measured numbers that separates the two kinds.

A law here has the same form as EPI: a burst is called Type I (a merger) when

    sum_i a_i log10(x_i) > c,

for a few numbers x_i (durations, MVTs, E_peaks) with exponents a_i in steps of 0.5, like EPI's 2, -1, -1/2 and -1. That keeps a law short
enough to write down and read physically. EPI is one such law, with four terms.

How laws are found and judged:
  - search() tries every set of up to max_terms numbers with every allowed exponent, and scores each by the area under the ROC curve
    (threshold-free: how well the score orders Type I above Type II).
  - nested_score() gives an honest success rate for the whole procedure: the burst sample is split into folds, the search is run on all
    folds but one, the best law and its threshold are taken from those, and the law is tried on the fold left out. Picking the best of
    thousands of laws on the same bursts used to judge it would flatter it; this doesn't.
  - probabilities() gives each burst a chance of being Type I under a law, from draws of its inputs within their errors (features.py).
The labels are Zhu et al.'s types, which come from clustering numbers that most laws also use, so agreement with them is partly built in
(notebook 22, section 8). The precursor's own E_peak is not among their numbers, so a law that needs it is a test they didn't make.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

NAMES = {"T100_PE": "T_PE", "T100_QE1": "T_QE1", "TMVT_PE": "MVT_PE", "Ep_ME_zhu": "Ep_ME", "Ep_PE_ours": "Ep_PE", "T100_ME": "T_ME",
         "TMVT_ME": "MVT_ME", "T100_WE": "T_WE"}
PRECURSOR_TIMING = ["T_PE", "T_QE1", "MVT_PE"]
PRECURSOR = PRECURSOR_TIMING + ["Ep_PE"]
EPI_INPUTS = PRECURSOR_TIMING + ["Ep_ME"]
EXPONENTS = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)


def aucs(S, y):
    """auc for every column of S at once."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    ranks = rankdata(S, axis=0)
    return (ranks[y].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc(score, y):
    """Area under the ROC curve of score for labels y (1 is Type I), by ranks, with ties counted half."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    ranks = pd.Series(score).rank().values
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def best_threshold(score, y):
    """The threshold c for 'score > c is Type I' with the best balanced accuracy (mean of the two classes' success rates), halfway between
    neighbouring scores. Returns (c, balanced accuracy)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    order = np.argsort(score)
    s, labels = score[order], y[order]
    n1, n0 = labels.sum(), (~labels).sum()
    # with the line just above position k, the bursts at 0..k are called Type II
    below_1 = np.cumsum(labels)
    below_0 = np.cumsum(~labels)
    balanced = 0.5 * ((n1 - below_1) / n1 + below_0 / n0)
    balanced = np.r_[0.5 * (1.0 + 0.0), balanced]               # the line below everything: all called Type I
    k = int(np.argmax(balanced))
    if k == 0:
        return s[0] - 1e-9, balanced[0]
    c = s[k - 1] if k == len(s) else 0.5 * (s[k - 1] + s[k])
    return c, balanced[k]


def candidate_laws(names, max_terms=3, exponents=EXPONENTS):
    """Every exponent vector on up to max_terms of the names, one of each pair a and -a (they rank the bursts in reverse), and none that
    is a multiple of another (2,1 is the same law as 1,0.5). Returns a list of dicts name -> exponent."""
    seen, laws = set(), []
    for k in range(1, max_terms + 1):
        for subset in combinations(names, k):
            for values in product(exponents, repeat=k):
                v = np.array(values)
                v = v / np.abs(v).max()
                if v[0] < 0:
                    v = -v
                key = (subset, tuple(np.round(v, 6)))
                if key in seen:
                    continue
                seen.add(key)
                laws.append(dict(zip(subset, values if values[0] > 0 else tuple(-x for x in values))))
    return laws


def scores_of(laws, X):
    """The score of every law for every burst: a (bursts x laws) array. X holds log10 of the numbers, one column per name."""
    names = list(X.columns)
    A = np.zeros((len(names), len(laws)))
    for j, law in enumerate(laws):
        for name, a in law.items():
            A[names.index(name), j] = a
    return X.values @ A


def search(X, y, max_terms=3, exponents=EXPONENTS, top=20):
    """Every law on the columns of X (log10 values, no missing ones), ranked by its area under the ROC curve on these bursts. The direction
    is set so that a larger score means Type I. Returns a table of the top laws with their AUC, threshold and balanced accuracy."""
    laws = candidate_laws(list(X.columns), max_terms, exponents)
    S = scores_of(laws, X)
    y = np.asarray(y, bool)
    areas = aucs(S, y)
    flip = areas < 0.5
    areas = np.where(flip, 1 - areas, areas)
    order = np.lexsort((np.array([len(l) for l in laws]), -areas))[:top]
    rows = []
    for j in order:
        law = {k: (-v if flip[j] else v) for k, v in laws[j].items()}
        score = S[:, j] * (-1 if flip[j] else 1)
        c, bal = best_threshold(score, y)
        rows.append(dict(law=law, text=describe(law), terms=len(law), auc=areas[j], threshold=c, balanced_accuracy=bal))
    return pd.DataFrame(rows)


def describe(law):
    """A law as text, like '2 log Ep_ME - 1 log T_PE - 0.5 log T_QE1'."""
    parts = []
    for i, (name, a) in enumerate(law.items()):
        sign = "-" if a < 0 else ("+" if i else "")
        parts.append(f"{sign} {abs(a):g} log {name}".strip())
    return " ".join(parts)


def apply(law, X):
    """The score of one law for the bursts in X."""
    return sum(a * X[name] for name, a in law.items())


def nested_score(X, y, max_terms=3, exponents=EXPONENTS, folds=5, repeats=10, seed=5):
    """The success rate of 'search, then use the best law' on bursts it hasn't seen: in each of folds x repeats splits the search and the
    threshold use the other folds only. Returns (table of the outer folds, the laws chosen and how often each was chosen)."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    y = np.asarray(y, bool)
    laws = candidate_laws(list(X.columns), max_terms, exponents)
    S = scores_of(laws, X)
    rows, chosen = [], []
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    for train, test in splitter.split(S, y):
        areas = aucs(S[train], y[train])
        flip = areas < 0.5
        areas = np.where(flip, 1 - areas, areas)
        j = int(np.lexsort((np.array([len(l) for l in laws]), -areas))[0])
        sign = -1 if flip[j] else 1
        c, _ = best_threshold(sign * S[train, j], y[train])
        called = sign * S[test, j] > c
        truth = y[test]
        rows.append(dict(balanced_accuracy=0.5 * (called[truth].mean() + (~called[~truth]).mean()), type_I_found=called[truth].mean(),
                         type_II_found=(~called[~truth]).mean(), auc_test=auc(sign * S[test, j], truth)))
        chosen.append(describe({k: sign * v for k, v in laws[j].items()}))
    return pd.DataFrame(rows), pd.Series(chosen).value_counts()


def law_threshold_bootstrap(law, X, y, boot=2000, seed=9):
    """The threshold of a fixed law refitted on bootstrap resamples of the bursts: its 16th, 50th and 84th percentiles."""
    rng = np.random.default_rng(seed)
    score, y = apply(law, X).values, np.asarray(y, bool)
    values = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and (~y[i]).any():
            values.append(best_threshold(score[i], y[i])[0])
    return np.percentile(values, [15.87, 50.0, 84.13])


def probabilities(law, threshold, draws):
    """Chance of each burst being called Type I by the law: the share of its draws scoring above the threshold. draws maps each burst
    to a DataFrame of draws of the law's numbers (log10 values), e.g. from features.input_draws."""
    out = {}
    for name, d in draws.items():
        if all(k in d for k in law) and len(d):
            out[name] = float(np.mean(apply(law, d) > threshold))
    return pd.Series(out, dtype=float)


def inputs(ours, measured_pe_only=True, way=None):
    """log10 of the numbers a law can use, one row per burst, named as in NAMES. E_p,ME is their way (Ep_ME_zhu) unless way="ours" or the
    table is from the project mode, whose E_peaks are all by our rule. The precursor's E_peak is kept only where our rule counts it as
    measured (otherwise it is missing, and a law that needs it skips that burst)."""
    if way is None:
        way = "ours" if "mode" in ours and (ours["mode"] == "project").all() else "zhu"
    names = {(f"Ep_ME_{way}" if long == "Ep_ME_zhu" else long): short for long, short in NAMES.items()}
    X = pd.DataFrame({short: np.log10(ours[long].astype(float)) for long, short in names.items() if long in ours})
    if measured_pe_only and "measured_PE" in ours:
        X.loc[ours.measured_PE.astype(str) != "True", "Ep_PE"] = np.nan
    return X


# ---------------------------------------------------------------------------------------------------------------------------------------
# Labels, and testing whether a number adds anything

# Origins confirmed by an associated kilonova or supernova. Zhu et al.'s types, and the short bursts in old hosts that they say are
# "generally considered" Type I-S (090510, 111117A, 161001A), are candidate labels: they are never used as if they were confirmed.
CONFIRMED = {"211211A": "merger", "230307A": "merger", "090618A": "collapsar", "111228A": "collapsar", "180728A": "collapsar",
             "190114C": "collapsar", "190829A": "collapsar", "211023A": "collapsar"}


def label_table(names, types):
    """For each burst: Zhu et al.'s type (a candidate label), whether it is confirmed, and the confirmed origin if so."""
    out = pd.DataFrame(index=pd.Index(names))
    out["zhu_type"] = types.reindex(out.index)
    out["candidate_merger"] = out.zhu_type.str.startswith("I-")
    out["confirmed"] = [CONFIRMED.get(n, "") for n in out.index]
    out["label_status"] = np.where(out.confirmed != "", "confirmed", "candidate")
    return out


def _cv_scores(X, y, folds, repeats, seed):
    """Logistic regression on standardised columns with classes weighted by their size, scored on the folds left out: area under the ROC
    curve, balanced accuracy at a probability of 0.5, average precision (the precision-recall area, which watches the rare class), and log
    loss. Returns one row per fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = np.asarray(X, float), np.asarray(y, bool)
    rows = []
    for train, test in RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000)).fit(X[train], y[train])
        p = model.predict_proba(X[test])[:, 1]
        rows.append(dict(auc=roc_auc_score(y[test], p), balanced_accuracy=balanced_accuracy_score(y[test], p > 0.5),
                         average_precision=average_precision_score(y[test], p), log_loss=log_loss(y[test], p, labels=[False, True])))
    return pd.DataFrame(rows)


def compare_feature_sets(X, y, sets, folds=5, repeats=20, seed=7):
    """Each set of columns scored on the same folds. Returns (summary: mean and spread of each score per set, and the paired difference of
    each set's log loss from the first set's, with how often it is better)."""
    scores = {name: _cv_scores(X[cols], y, folds, repeats, seed) for name, cols in sets.items()}
    base = list(sets)[0]
    rows = []
    for name, s in scores.items():
        diff = s.log_loss - scores[base].log_loss
        rows.append(dict(features=name, **{f"{k}_mean": s[k].mean() for k in s}, **{f"{k}_spread": s[k].std() for k in s},
                         log_loss_vs_first=diff.mean(), better_than_first=float(np.mean(diff < 0))))
    return pd.DataFrame(rows).set_index("features"), scores


def permutation_test(X, y, base_cols, extra, permutations=200, folds=5, repeats=4, seed=11):
    """Does the column `extra` add to base_cols? The cross-validated log loss with it is compared with the same after shuffling `extra`
    among the bursts, which keeps its spread but breaks any link with the labels. Returns (observed gain, the gains with it shuffled, p): p
    is the share of shuffles that gain as much as the real column, so a small p means the column carries information about the labels."""
    rng = np.random.default_rng(seed)
    base = _cv_scores(X[base_cols], y, folds, repeats, seed).log_loss.mean()
    observed = base - _cv_scores(X[base_cols + [extra]], y, folds, repeats, seed).log_loss.mean()
    gains = []
    for _ in range(permutations):
        shuffled = X[base_cols + [extra]].copy()
        shuffled[extra] = rng.permutation(shuffled[extra].values)
        gains.append(base - _cv_scores(shuffled, y, folds, repeats, seed).log_loss.mean())
    gains = np.array(gains)
    return observed, gains, float((np.sum(gains >= observed) + 1) / (len(gains) + 1))


def bootstrap_auc(score, y, boot=2000, seed=3):
    """The area under the ROC curve of a fixed score, with a bootstrap 68% and 95% interval over bursts."""
    rng = np.random.default_rng(seed)
    score, y = np.asarray(score, float), np.asarray(y, bool)
    values = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and (~y[i]).any():
            values.append(auc(score[i], y[i]))
    return auc(score, y), np.percentile(values, [15.87, 84.13]), np.percentile(values, [2.5, 97.5])
