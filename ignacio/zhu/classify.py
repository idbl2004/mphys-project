"""Their classification repeated on their own table, then tested; and EPI's threshold with an uncertainty.

Zhu et al. 2026 (sections 2.3 and 3): nine numbers per burst (T100 of PE, ME, WE and QE1, T_MVT of PE and ME, tau32 of ME and WE, E_p,ME),
log10 of all but the two lags, then z-scored; t-SNE (perplexity 6) and UMAP (6 neighbours, min_dist 1e-4) down to two dimensions; HDBSCAN
from scikit-learn on each map. They name the clusters by the bursts of known origin in them: the cluster with the kilonova bursts is
Type I-L, the one with the supernova bursts Type II, and the one with three short bursts in old hosts (090510, 111117A, 161001A) Type I-S.
Their Type I-L are the Type I with T100,ME > 2 s and T100,WE > 10 s.

Here the same steps run on their table, with each random seed and setting recorded, and then:
  - many seeds and settings, to see how often the merger group comes back the same;
  - clustering in the nine-dimensional space itself (no map), with Gaussian mixtures of three kinds and with HDBSCAN;
  - which numbers carry the split: the separation of each one alone, and the map with each one left out;
  - groups made only from the five numbers that are not in EPI, to see how much of the split those numbers carry on their own;
  - the EPI line with a bootstrap interval, and the chance a burst with a measured EPI and error lies above it.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from .catalogue import EPI_LINE

FEATURES = ["T100_PE", "T100_ME", "T100_WE", "T100_QE1", "TMVT_PE", "TMVT_ME", "tau32_ME_ms", "tau32_WE_ms", "Ep_ME"]
LINEAR = ["tau32_ME_ms", "tau32_WE_ms"]                 # they can be negative, so they aren't logged
EPI_INPUTS = ["Ep_ME", "T100_PE", "T100_QE1", "TMVT_PE"]
KILONOVA_ANCHORS = ["211211A", "230307A", "060614A"]
SHORT_HOST_ANCHORS = ["090510A", "111117A", "161001A"]
SUPERNOVA_ANCHORS = ["090618A", "111228A", "180728A", "190114C", "190829A", "211023A"]
SETTINGS = dict(perplexity=6, n_neighbors=6, min_dist=1e-4, min_cluster_size=5)


def design_matrix(table, features=FEATURES):
    """log10 of the positive numbers, the lags as they are, each column z-scored, as they describe."""
    x = pd.DataFrame({f: (table[f] if f in LINEAR else np.log10(table[f])) for f in features}, index=table.index)
    return (x - x.mean()) / x.std(ddof=0)


def embed(x, method, seed, perplexity=SETTINGS["perplexity"], n_neighbors=SETTINGS["n_neighbors"], min_dist=SETTINGS["min_dist"], init="pca"):
    """A two-dimensional map of the rows of x with t-SNE or UMAP. t-SNE starts from the first two principal components by default, as
    scikit-learn does, which makes it the same for every seed; init='random' lets the seed matter."""
    if method == "tsne":
        return TSNE(n_components=2, perplexity=perplexity, random_state=seed, init=init).fit_transform(np.asarray(x))
    import umap                                           # slow to import, so only when it's used
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(np.asarray(x))


def clusters(points, min_cluster_size=SETTINGS["min_cluster_size"]):
    """HDBSCAN labels of points (scikit-learn, as they used); -1 is noise."""
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(np.asarray(points))


def name_clusters(labels, names):
    """Their naming, as it works out on their own table: a cluster holding a kilonova or old-host short burst and no supernova burst is
    'merger'; one holding a supernova burst and none of the others is 'collapsar'; every other cluster, and the noise, is 'other'.

    With scikit-learn's default HDBSCAN (min_cluster_size 5) the t-SNE map of their table splits into about 26 clusters, and the ones with
    merger anchors hold exactly their 44 Type I bursts, while only about a sixth of their Type II sit in clusters with a supernova burst.
    So their Type II is in practice "not in a merger cluster", and compare() scores it that way. Returns a Series indexed by burst."""
    names = pd.Index(names)
    merger_anchor = names.isin(KILONOVA_ANCHORS + SHORT_HOST_ANCHORS)
    sn_anchor = names.isin(SUPERNOVA_ANCHORS)
    out = pd.Series("other", index=names)
    for c in set(labels) - {-1}:
        inside = labels == c
        has_m, has_s = merger_anchor[inside].any(), sn_anchor[inside].any()
        if has_m and not has_s:
            out[inside] = "merger"
        elif has_s and not has_m:
            out[inside] = "collapsar"
    return out


def compare(named, types):
    """Agreement of a naming with their Types, scoring 'merger' against everything else: the share of their Type I called merger, the
    share of their Type II not called merger, the adjusted Rand index, and the share of their Type II that sits in a supernova cluster."""
    merger_theirs = types.str.startswith("I-")
    ours = named == "merger"
    return dict(type_I_recovered=float(ours[merger_theirs].mean()), type_II_recovered=float((~ours[~merger_theirs]).mean()),
                ari=adjusted_rand_score(merger_theirs, ours), type_II_in_sn_cluster=float((named[~merger_theirs] == "collapsar").mean()))


def one_run(x, types, method, seed, min_cluster_size=SETTINGS["min_cluster_size"], **settings):
    """Map, clusters, names and agreement for one seed and setting. Returns (row, labels)."""
    points = embed(x, method, seed, **settings)
    labels = clusters(points, min_cluster_size)
    named = name_clusters(labels, x.index)
    row = dict(method=method, seed=seed, min_cluster_size=min_cluster_size, **settings, **compare(named, types))
    row["n_clusters"] = int(len(set(labels) - {-1}))
    row["noise"] = float(np.mean(labels == -1))
    return row, labels


def scan(x, types, method, values, seeds, key):
    """one_run over a grid of one setting and seeds. Returns a table, one row per run."""
    rows = []
    for v in values:
        for seed in seeds:
            row, _ = one_run(x, types, method, seed, **{key: v})
            rows.append(row)
    return pd.DataFrame(rows)


def robustness(table, seeds=range(20)):
    """Their steps over seeds and settings, one row per run: t-SNE over perplexity (from principal components, and from random starts with
    each seed), UMAP over n_neighbors and seeds at their min_dist and at 0.1, and both over HDBSCAN's min_cluster_size at their map settings."""
    x = design_matrix(table)
    parts = []
    for perplexity in (3, 4, 5, 6, 8, 10, 15, 20, 30, 50):
        parts.append(pd.DataFrame([one_run(x, table.Type, "tsne", 0, perplexity=perplexity)[0]]).assign(init="pca"))
        parts.append(pd.DataFrame([one_run(x, table.Type, "tsne", s, perplexity=perplexity, init="random")[0] for s in seeds]).assign(init="random"))
    for min_dist in (SETTINGS["min_dist"], 0.1):
        for n_neighbors in (3, 4, 5, 6, 8, 10, 15, 20, 30, 50):
            parts.append(pd.DataFrame([one_run(x, table.Type, "umap", s, n_neighbors=n_neighbors, min_dist=min_dist)[0] for s in seeds]))
    for size in (3, 4, 5, 6, 8, 10, 15, 20):
        parts.append(pd.DataFrame([one_run(x, table.Type, "tsne", 0, min_cluster_size=size)[0]]).assign(init="pca", scan="min_cluster_size"))
        parts.append(pd.DataFrame([one_run(x, table.Type, "umap", s, min_cluster_size=size)[0] for s in seeds]).assign(scan="min_cluster_size"))
    return pd.concat(parts, ignore_index=True)


def mixture_in_feature_space(x, types, max_components=6, seeds=range(10), covariances=("full", "tied", "diag")):
    """Gaussian mixtures on the nine numbers themselves, 1 to max_components, each the best of several seeds by BIC, for each kind of
    covariance (full: each component its own; tied: one shared; diag: no correlations). For each, the component holding the most of their
    Type I: its size, how many of their Type I and Type II it holds, and the adjusted Rand index of 'in that component' against their
    Type I. Returns one row per covariance kind and number of components."""
    merger_theirs = types.str.startswith("I-").values
    rows = []
    for covariance in covariances:
        for k in range(1, max_components + 1):
            best = min((GaussianMixture(k, covariance_type=covariance, random_state=s).fit(x) for s in seeds), key=lambda g: g.bic(x))
            labels = best.predict(x)
            counts = [(merger_theirs[labels == c].sum(), c) for c in range(k)]
            _, top = max(counts)
            inside = labels == top
            rows.append(dict(covariance=covariance, components=k, bic=best.bic(x), sizes=" ".join(str(int(np.sum(labels == c))) for c in range(k)),
                             top_size=int(inside.sum()), top_type_I=int(merger_theirs[inside].sum()), top_type_II=int((~merger_theirs[inside]).sum()),
                             ari=adjusted_rand_score(merger_theirs, inside)))
    return pd.DataFrame(rows)


def hdbscan_in_feature_space(x, types, sizes=(4, 5, 6, 8, 10, 15, 20)):
    """HDBSCAN on the nine numbers themselves, with no map, named by the same anchors as the maps. One row per min_cluster_size."""
    rows = []
    for size in sizes:
        labels = clusters(x.values, size)
        named = name_clusters(labels, x.index)
        rows.append(dict(min_cluster_size=size, n_clusters=int(len(set(labels) - {-1})), noise=float(np.mean(labels == -1)), **compare(named, types)))
    return pd.DataFrame(rows)


def single_feature_separation(table, features=FEATURES + ["T90", "Tres", "EPI"]):
    """Area under the ROC curve of each number alone for their Type I against Type II (0.5 is no separation, 1 is complete). For numbers
    that are smaller in Type I the area is turned round, so every value is at least 0.5."""
    merger = table.Type.str.startswith("I-")
    rows = []
    for f in features:
        auc = roc_auc_score(merger, table[f])
        rows.append(dict(feature=f, auc=max(auc, 1 - auc), larger_in_type_I=auc >= 0.5))
    return pd.DataFrame(rows).set_index("feature").sort_values("auc", ascending=False)


def leave_one_out(table, method="umap", seeds=range(20)):
    """The map and clusters without each of the nine numbers in turn: agreement with their types over seeds (mean, spread, lowest), and
    how often the merger group comes back exactly."""
    rows = []
    for dropped in [None] + FEATURES:
        keep = [f for f in FEATURES if f != dropped]
        x = design_matrix(table, keep)
        runs = pd.DataFrame([one_run(x, table.Type, method, s)[0] for s in seeds])
        rows.append(dict(left_out=dropped or "none", ari_mean=runs.ari.mean(), ari_spread=runs.ari.std(), ari_lowest=runs.ari.min(),
                         exact=float((runs.ari > 0.999).mean()), type_I_recovered=runs.type_I_recovered.mean(), n_clusters=runs.n_clusters.median()))
    return pd.DataFrame(rows).set_index("left_out")


def without_epi_inputs(table, method="umap", seeds=range(10), min_cluster_size=SETTINGS["min_cluster_size"]):
    """Groups made only from the five numbers that are not in EPI (T100,ME, T100,WE, T_MVT,ME and the two lags), named by the same anchors,
    against EPI on either side of 6.2 and against their types. The groups share no column with EPI, but they do share information:
    T100,WE is T100,PE + T100,QE1 + T100,ME for every burst without extended emission, and tau32,WE is measured over the whole burst. So
    this measures how much of the split those five numbers carry, not how circular EPI is."""
    rest = [f for f in FEATURES if f not in EPI_INPUTS]
    x = design_matrix(table, rest)
    above = table.EPI > EPI_LINE
    rows = []
    for seed in seeds:
        points = embed(x, method, seed)
        named = name_clusters(clusters(points, min_cluster_size), x.index)
        rows.append(dict(seed=seed, n_merger=int((named == "merger").sum()), ari_with_epi=adjusted_rand_score(above, named == "merger"),
                         epi_agrees=float(((named == "merger") == above).mean()),
                         ari_with_types=adjusted_rand_score(table.Type.str.startswith("I-"), named == "merger")))
    return pd.DataFrame(rows)


def epi_threshold(table, boot=2000, seed=2026):
    """The EPI value that best separates their Type I from Type II (largest true-positive minus false-positive rate), with a bootstrap
    interval over bursts. Returns (best, 16th, 84th percentile, all the bootstrap values)."""
    merger = table.Type.str.startswith("I-").values
    epi = table.EPI.values

    def best(m, e):
        fpr, tpr, thr = roc_curve(m, e)
        k = int(np.argmax(tpr - fpr))
        # halfway between this value and the next one down, so the line sits in the gap rather than on a burst
        below = e[e < thr[k]]
        return 0.5 * (thr[k] + below.max()) if below.size else thr[k]

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(boot):
        i = rng.integers(0, len(epi), len(epi))
        if merger[i].any() and (~merger[i]).any():
            values.append(best(merger[i], epi[i]))
    values = np.array(values)
    return best(merger, epi), *np.percentile(values, [15.87, 84.13]), values


def probability_above(epi_value, epi_lo, epi_hi, line=EPI_LINE):
    """Chance the true EPI is above the line, for a measured EPI with uneven minus and plus errors.

    The measurement is read as a two-piece normal: width epi_lo below the value and epi_hi above it, with the two halves joined at the value
    (so a share epi_hi / (epi_lo + epi_hi) of the probability lies above it)."""
    s1, s2 = epi_lo, epi_hi
    if not (np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 > 0):
        return float(epi_value > line)
    if line >= epi_value:
        return float(2 * s2 / (s1 + s2) * norm.sf((line - epi_value) / s2))
    return float(1 - 2 * s1 / (s1 + s2) * norm.cdf((line - epi_value) / s1))
