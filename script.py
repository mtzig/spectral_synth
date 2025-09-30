# quick_fiedler_mixture_ms_marco_tqdm.py
# Usage:
#   python quick_fiedler_mixture_ms_marco_tqdm.py --limit 20000 --k 15 \
#       --mixtures 0.0 0.1 0.2 0.4 0.6 0.8 1.0 --mode augment
#
# Dependencies: datasets, numpy, scikit-learn, scipy, sentence-transformers, torch, tqdm

import argparse
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lobpcg, eigsh, LinearOperator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_ms_marco_queries(limit=None, shuffle=True, seed=42, prefer_config="v1.1"):
    """
    Loads MS MARCO (passage) queries from the official HF dataset.

    Tries microsoft/ms_marco with config v1.1 (preferred), then v2.1.
    Returns:
        queries: list[str]
        qids:    list[str]
    """
    last_err = None
    for cfg in [prefer_config, "v1.1" if prefer_config != "v1.1" else "v2.1", "v2.1"]:
        try:
            ds = load_dataset("microsoft/ms_marco", cfg, split="train")
            # Expected fields on v1.1/v2.1: 'query', 'query_id', 'passages' (we only need the first two)
            # But we’ll be robust to alternative names just in case.
            possible_q_text = ["query", "query_text"]
            possible_qid = ["query_id", "qid", "id"]

            # find fields
            q_text_field = next((f for f in possible_q_text if f in ds.features), None)
            qid_field = next((f for f in possible_qid if f in ds.features), None)
            if q_text_field is None or qid_field is None:
                raise ValueError(f"Unexpected fields in microsoft/ms_marco:{cfg} -> {list(ds.features.keys())}")

            queries = ds[q_text_field]
            qids = [str(x) for x in ds[qid_field]]

            idxs = np.arange(len(queries))
            if shuffle:
                rng = np.random.default_rng(seed)
                rng.shuffle(idxs)
            if limit:
                idxs = idxs[:limit]
            queries = [queries[i] for i in idxs]
            qids = [qids[i] for i in idxs]
            return queries, qids
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Failed to load microsoft/ms_marco with configs ['v1.1','v2.1'].\n"
        f"Last error: {last_err}"
    )

def load_query2doc_expansions(use_splits=("train",), repo_id="intfloat/query2doc_msmarco", revision="main"):
    """
    Returns:
        dict[str, list[str]] mapping query_id -> list of pseudo_doc expansions.
    Notes:
        - Loads JSONL files directly from the dataset repo (no loading script).
        - Available files: dev.jsonl, test.jsonl, train.jsonl, trec_dl2019.jsonl, trec_dl2020.jsonl.
    """
    local_files = {}
    for split in use_splits:
        filename = f"{split}.jsonl"
        local_files[split] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",   # <-- key fix
            revision=revision      # optional pin; keep "main" or a commit SHA
        )

    datasets_list = []
    for path in local_files.values():
        ds = load_dataset("json", data_files=path, split="train")  # single-file “train” split
        datasets_list.append(ds)
    ds_all = datasets_list[0] if len(datasets_list) == 1 else concatenate_datasets(datasets_list)

    # Expected columns in this repo
    if not {"query_id", "pseudo_doc"}.issubset(set(ds_all.column_names)):
        raise ValueError(f"Unexpected columns: {ds_all.column_names}")

    exp_by_qid = defaultdict(list)
    for ex in ds_all:
        exp_by_qid[str(ex["query_id"])].append(ex["pseudo_doc"])
    return exp_by_qid


def make_mixture(real_qs, real_qids, exp_by_qid, alpha, mode="augment", per_real=1, seed=42):
    """
    Build a list of texts for a given synthetic mixture.

    Args:
        real_qs, real_qids: lists of equal length
        exp_by_qid: dict qid -> list of synthetic strings
        alpha: fraction of synthetic in the final set (0..1)
        mode: 'augment' or 'replace'
        per_real: number of synthetic queries to sample per real (used in 'replace')
    Returns:
        texts: list[str]
    """
    rng = np.random.default_rng(seed)
    n_real = len(real_qs)

    if mode == "augment":
        # alpha = S / (R + S)  ->  S = alpha/(1-alpha) * R
        if alpha >= 1.0:
            s_target = n_real * 4
        elif alpha <= 0.0:
            s_target = 0
        else:
            s_target = int(round((alpha / (1.0 - alpha)) * n_real))

        synth = []
        qid_order = np.arange(n_real)
        rng.shuffle(qid_order)
        i = 0
        # tqdm cosmetic loop to show we're sampling synthetics
        with tqdm(total=max(s_target, 1), desc="Sampling synthetic (augment)", leave=False) as pbar:
            while len(synth) < s_target and i < 10 * n_real:
                qid = real_qids[qid_order[i % n_real]]
                if qid in exp_by_qid and len(exp_by_qid[qid]) > 0:
                    pick = rng.choice(exp_by_qid[qid])
                    synth.append(pick)
                    pbar.update(1)
                i += 1

        texts = list(real_qs) + synth

        # temp hack to test stuff
        # texts = synth
        return texts

    elif mode == "replace":
        n_replace = int(round(alpha * n_real))
        replace_idx = rng.choice(np.arange(n_real), size=n_replace, replace=False)
        replace_mask = np.zeros(n_real, dtype=bool)
        replace_mask[replace_idx] = True

        texts = []
        for i, (q, qid) in enumerate(tqdm(zip(real_qs, real_qids), total=n_real, desc="Replacing (replace)", leave=False)):
            if not replace_mask[i]:
                texts.append(q)
            else:
                if qid in exp_by_qid and len(exp_by_qid[qid]) > 0:
                    picks = rng.choice(exp_by_qid[qid], size=min(per_real, len(exp_by_qid[qid])), replace=True)
                    texts.extend(list(picks))
                else:
                    texts.append(q)
        return texts
    else:
        raise ValueError("mode must be 'augment' or 'replace'")


def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=512, device=None):
    """
    Batched embedding with a tqdm progress bar.
    """
    model = SentenceTransformer(model_name, device=device)
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i:i + batch_size]
        e = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embs.append(e)
    return np.vstack(embs)


def knn_graph(embeddings, k=15, metric="cosine"):
    """
    Symmetric sparse adjacency via kNN (cosine similarity).
    """
    n = embeddings.shape[0]
    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", metric=metric)
    nbrs.fit(embeddings)

    # kneighbors is one-shot; we wrap with a tiny tqdm step to show progress statefully
    with tqdm(total=1, desc="kNN search", leave=False):
        distances, indices = nbrs.kneighbors(embeddings)

    rows, cols, vals = [], [], []
    for i in range(n):
        for d, j in zip(distances[i][1:], indices[i][1:]):  # skip self
            sim = 1.0 - d
            if sim < 0:
                sim = 0.0
            rows.append(i); cols.append(j); vals.append(sim)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetrize
    return A


def normalized_laplacian(A):
    d = np.array(A.sum(axis=1)).ravel()
    d_safe = np.where(d > 0, d, 1.0)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_safe))
    n = A.shape[0]
    I = sparse.identity(n, format="csr")
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L


# def fiedler_value(L):
#     """
#     Second-smallest eigenvalue of the normalized Laplacian.
#     """
#     # Cosmetic tqdm wrapper since eigsh is one-shot
#     with tqdm(total=1, desc="Eigen (eigsh)", leave=False):
#         vals, _ = eigsh(L, k=2, which="SM", tol=1e-3)
#     vals = np.sort(vals)
#     if len(vals) < 2:
#         return float("nan")
#     return float(vals[1])

def _largest_component_laplacian(L, degrees):
    # Keep largest connected component of the underlying graph (A from L = I - D^-1/2 A D^-1/2)
    # We infer adjacency pattern from L: A ~ I - L scaled, but topology (nonzeros) matches off-diagonals.
    # Build an unweighted adjacency from L's off-diagonals:
    n = L.shape[0]
    # Off-diagonal structure: edges where L[i,j] != 0 and i!=j
    Lcoo = L.tocoo()
    mask = Lcoo.row != Lcoo.col
    rows = Lcoo.row[mask]; cols = Lcoo.col[mask]
    Adj = sparse.csr_matrix((np.ones_like(rows, dtype=np.uint8), (rows, cols)), shape=(n, n))
    Adj = Adj.maximum(Adj.T)  # ensure symmetric

    # Remove isolated nodes early (degree 0 in Adj)
    deg = np.array(Adj.sum(axis=1)).ravel()
    keep = deg > 0
    if keep.sum() == 0:
        # Graph has no edges; λ2 is undefined -> return NaN
        return None, None, None

    Adj = Adj[keep][:, keep]
    # connected components on pruned graph
    n_comp, labels = connected_components(Adj, directed=False)
    if n_comp <= 1:
        idx = np.where(keep)[0]
        return L[idx[:,None], idx], None, idx

    # pick largest component
    sizes = np.bincount(labels)
    comp_id = sizes.argmax()
    sel = (labels == comp_id)
    idx_keep = np.where(keep)[0][sel]
    L_cc = L[idx_keep[:,None], idx_keep]
    return L_cc, comp_id, idx_keep

def fiedler_value_fast(L, x0=None, use_amg=True, use_shift_invert=False, tol=1e-3, maxiter=200):
    """
    Compute λ2 (Fiedler) of a (normalized) Laplacian L (sparse CSR/CSC).
    - x0: warm-start vector (shape [n,1]); pass previous run's Fiedler eigenvector when sweeping α.
    - use_amg: try PyAMG preconditioner for LOBPCG (if available).
    - use_shift_invert: if True, ARPACK shift-invert with sigma=0 (requires sparse factorization).
    """
    n = L.shape[0]
    # Build degrees to identify isolated nodes and components
    # For normalized Laplacian, D = I on diagonal of L? Not exactly. Use adjacency pattern instead.
    degrees = None  # not strictly needed below

    # Work on largest CC (and drop isolated nodes)
    L_cc, _, idx = _largest_component_laplacian(L, degrees)
    if L_cc is None:
        return np.nan, None  # no edges

    m = L_cc.shape[0]
    if m < 2:
        return np.nan, None

    # Warm-start: random if not provided or wrong size
    if x0 is None or (idx is not None and x0.shape[0] != m) or (idx is None and x0 is not None and x0.shape[0] != n):
        rng = np.random.default_rng(0)
        x0_cc = rng.standard_normal((m, 1))
    else:
        # map x0 to the CC subspace if needed
        x0_cc = x0 if (idx is None or x0.shape[0] == m) else x0[idx][:, None] if x0.ndim == 1 else x0[idx]

    # Try LOBPCG + AMG preconditioner
    if use_amg:
        try:
            import pyamg
            # Build a symmetric positive semi-definite preconditioner
            # For normalized Laplacian, AMG works well.
            ml = pyamg.ruge_stuben_solver(L_cc.tocsr())
            M = ml.aspreconditioner(cycle='V')
            # Compute the 2 smallest eigenpairs; we’ll drop the first (≈0) and keep λ2
            w, V = lobpcg(L_cc, X=x0_cc, M=M, tol=tol, maxiter=maxiter, largest=False, k=2)
            w = np.sort(w)
            if len(w) >= 2:
                return float(w[1]), (V[:, np.argsort(w)[1]].reshape(-1, 1))
        except Exception:
            pass  # fall back below

    # Fall back: ARPACK
    try:
        if use_shift_invert:
            # Shift-invert targets smallest via sigma=0
            w, V = eigsh(L_cc, k=2, sigma=0.0, which="LM", tol=tol, maxiter=maxiter)
        else:
            # Without shift-invert, prefer 'SA' for symmetric PSD
            w, V = eigsh(L_cc, k=2, which="SA", tol=tol, maxiter=maxiter, ncv=max(20, 9))
        idx_sort = np.argsort(w)
        w = w[idx_sort]; V = V[:, idx_sort]
        return float(w[1]), (V[:, [1]])
    except Exception:
        # Last resort: your original method (may be slower)
        w, V = eigsh(L_cc, k=2, which="SM", tol=tol)
        w = np.sort(w)
        return float(w[1]), None

def fiedler_value(L, **kwargs):
    lam2, _ = fiedler_value_fast(L, **kwargs)
    return lam2

def run(limit, mixtures, k, mode, per_real, seed, model_name, device, batch_size):
    set_seed(seed)

    print("Loading MS MARCO queries…")
    real_qs, real_qids = load_ms_marco_queries(limit=limit, shuffle=True, seed=seed)

    print("Loading synthetic expansions (query2doc_msmarco)…")
    exp_by_qid = load_query2doc_expansions(use_splits=("train",))

    results = []
    for alpha in tqdm(mixtures, desc="Mixtures"):
        print(f"\n--- Mixture α={alpha:.2f} (mode={mode}) ---")
        texts = make_mixture(real_qs, real_qids, exp_by_qid, alpha=alpha, mode=mode, per_real=per_real, seed=seed)

        if mode == "augment":
            approx_synth = len(texts) - len(real_qs)
            print(f"Mixture size: {len(texts)} (real={len(real_qs)}, synthetic≈{approx_synth})")
        else:
            print(f"Mixture size: {len(texts)} (replace mode; size may vary)")

        X = embed_texts(texts, model_name=model_name, batch_size=batch_size, device=device)
        A = knn_graph(X, k=k, metric="cosine")
        L = normalized_laplacian(A)
        lam2 = fiedler_value(L)
        results.append((alpha, lam2))
        print(f"λ₂ (Fiedler) = {lam2:.6f}")

    print("\n=== Summary ===")
    for alpha, lam2 in results:
        print(f"alpha={alpha:.2f}\tFiedler={lam2:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20000, help="Number of real MS MARCO queries to sample.")
    parser.add_argument("--k", type=int, default=15, help="k for k-NN graph.")
    parser.add_argument("--mixtures", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], help="List of alpha values (0..1).")
    parser.add_argument("--mode", type=str, choices=["augment", "replace"], default="augment",
                        help="'augment' keeps all real and adds synthetic until S/(R+S)=alpha; 'replace' swaps a fraction alpha of real with synthetic.")
    parser.add_argument("--per_real", type=int, default=1, help="For 'replace' mode, how many synthetic items to sample per replaced real.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default=None, help="e.g., 'cuda', 'cuda:0', or leave None to auto")
    parser.add_argument("--batch_size", type=int, default=512, help="Embedding batch size.")
    args = parser.parse_args()

    run(
        limit=args.limit,
        mixtures=args.mixtures,
        k=args.k,
        mode=args.mode,
        per_real=args.per_real,
        seed=args.seed,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
