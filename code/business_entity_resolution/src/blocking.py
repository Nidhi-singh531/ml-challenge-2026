"""Candidate generation (blocking).

Design, driven by what the training data actually shows:

* ``country`` agrees on 100% of true pairs -> hard partition, zero recall cost.
* 100% of true pairs share >= 1 token between (name + address); 99.7% share >= 2.
  Requiring two shared tokens is therefore a near-free ~50x reduction.
* Rare-token-only blocking (df <= 5000) recalls just 92.6% — many true pairs
  share only a common city/state plus a common name word. So we index *all*
  tokens below a large df cap and rank by summed IDF instead of filtering hard.

Two channels, each an IDF-weighted inverted index with its own top-K, unioned:

* ``all``  - name + address tokens pooled (the original design). Finds pairs
  whose name was replaced or mangled but whose address survives.
* ``name`` - name tokens, concatenation keys and romanised consonant skeletons
  (``ertext.name_keys``). Measured on the full 10M-record pool, most links the
  ``all`` channel misses have an intact name but an empty or truncated address:
  a long Source-1 address lets hundreds of neighbours outscore them. A separate
  name top-K cannot be crowded out that way, and its keys also reach web-ified
  ('securecloudservices.com') and other-script names.

Scale (10M targets, 1.7-2.2M queries)
-------------------------------------
* Every record is tokenised exactly once; ``--cache`` saves the token ids so
  re-runs with other K / caps skip the ~6 minute tokenisation. The cached
  per-record token lists double as the forward index.
* Two tiers of tokens. *Generating* tokens (df <= ``--gen-df-cap``) retrieve
  candidates through the inverted index, batched as one sparse product
  ``Q @ P``. *Common* tokens (gen cap < df <= ``--df-cap``: city, state,
  'private') are too expensive to scan - one of them has 100k+ postings - so
  they only *add score* to candidates already generated, via the forward index.
* Score and shared-token count come out of one product by giving each query
  token the weight ``SHARED_UNIT + idf^2``: the integer part counts shared
  tokens, the remainder is the IDF score.

Outputs
-------
``<out>.tsv``         the submission-format candidate list (skip with --no-tsv)
``<out>_scores.npz``  one row per pair, ordered by entity then ``all`` score:
                      s1 / cand (int64-encoded ids, see ``encode_id``); score,
                      shared, rank (``all`` channel, recomputed for every pair of
                      the union); nscore, nshared, nrank (``name`` channel; nrank
                      is 999 when the pair did not make the name top-K); plus
                      ``queried``, every Source-1 id that was looked up.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from array import array

import numpy as np
from scipy import sparse

import ertext

SHARED_UNIT = float(2 ** 20)
_SRC_BASE = 1_000_000_000  # every id is S<k>-<int below 1e9>
CHANNELS = ("all", "name")
NO_RANK = 999


def encode_id(rid: str) -> int:
    """'S3-775321672' -> 3_775_321_672. Lossless for every id in the data."""
    return int(rid[1]) * _SRC_BASE + int(rid[3:])


def decode_id(code) -> str:
    code = int(code)
    return f"S{code // _SRC_BASE}-{code % _SRC_BASE}"


def iter_source(path):
    """Yield (id_str, name, address, country) from a source TSV."""
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts += [""] * (4 - len(parts))
            yield parts[0], parts[1], parts[2], parts[3]


def record_tokens(name, addr, amap=None):
    """Token multiset of the ``all`` channel: name tokens + address tokens.

    Passing the mined alias lexicon here is what lets a Devanagari record and
    its Latin counterpart land in the same blocks at all.
    """
    return ertext.canonical(ertext.name_tokens(name) + ertext.addr_tokens(addr),
                            amap)


def channel_keys(name, addr, amap):
    return (record_tokens(name, addr, amap), ertext.name_keys(name, amap))


# --------------------------------------------------------------------------- #
# tokenisation (once) + cache
# --------------------------------------------------------------------------- #
class _Bucket:
    """Growing CSR of token-id sets, one row per record."""

    def __init__(self):
        self.codes = array("q")
        self.offsets = array("q", [0])
        self.toks = array("i")

    def add(self, code, tids):
        self.codes.append(code)
        self.toks.extend(tids)
        self.offsets.append(len(self.toks))

    def arrays(self):
        return (np.frombuffer(self.codes, np.int64),
                np.frombuffer(self.offsets, np.int64),
                np.frombuffer(self.toks, np.int32))


def tokenize_all(data_dir, prefix, amap, t0):
    """-> (targets, queries, n_vocab); targets/queries are
    {(country, channel): (codes, offsets, toks)}, n_vocab {channel: size}."""
    vocab = {ch: {} for ch in CHANNELS}
    tgt, qry = {}, {}

    def bucket(store, ctry, ch):
        b = store.get((ctry, ch))
        if b is None:
            b = store[(ctry, ch)] = _Bucket()
        return b

    n = 0
    for i in (2, 3):
        for rid, name, addr, ctry in iter_source(
                os.path.join(data_dir, f"{prefix}_source{i}.tsv")):
            code = encode_id(rid)
            for ch, keys in zip(CHANNELS, channel_keys(name, addr, amap)):
                v = vocab[ch]
                bucket(tgt, ctry, ch).add(
                    code, sorted({v.setdefault(t, len(v)) for t in keys}))
            n += 1
            if n % 2_000_000 == 0:
                print(f"[blocking] tokenised {n} targets ({time.time() - t0:.0f}s)",
                      flush=True)
    for rid, name, addr, ctry in iter_source(
            os.path.join(data_dir, f"{prefix}_source1.tsv")):
        code = encode_id(rid)
        for ch, keys in zip(CHANNELS, channel_keys(name, addr, amap)):
            v = vocab[ch]
            bucket(qry, ctry, ch).add(code, sorted({v[t] for t in keys if t in v}))
    return ({k: b.arrays() for k, b in tgt.items()},
            {k: b.arrays() for k, b in qry.items()},
            {ch: len(v) for ch, v in vocab.items()})


def save_cache(path, tgt, qry, n_vocab):
    d = {f"n_vocab|{ch}": n for ch, n in n_vocab.items()}
    for kind, dd in (("t", tgt), ("q", qry)):
        for (c, ch), (codes, off, toks) in dd.items():
            p = f"{kind}|{c}|{ch}"
            d[p + "|codes"], d[p + "|off"], d[p + "|toks"] = codes, off, toks
    np.savez(path, **d)


def load_cache(path):
    d = np.load(path)
    tgt, qry, n_vocab = {}, {}, {}
    for k in d.files:
        parts = k.split("|")
        if parts[0] == "n_vocab":
            n_vocab[parts[1]] = int(d[k])
            continue
        kind, c, ch, part = parts
        (tgt if kind == "t" else qry).setdefault((c, ch), {})[part] = d[k]
    unpack = lambda dd: {k: (v["codes"], v["off"], v["toks"]) for k, v in dd.items()}
    return unpack(tgt), unpack(qry), n_vocab


# --------------------------------------------------------------------------- #
# index + query
# --------------------------------------------------------------------------- #
def _gather(offsets, toks, rows):
    """Token ids of the given records -> (row position, token id) arrays."""
    starts = offsets[rows]
    lens = offsets[rows + 1] - starts
    total = int(lens.sum())
    pos = np.repeat(np.arange(len(rows)), lens)
    within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    return pos, toks[np.repeat(starts, lens) + within]


class TargetIndex:
    """Two-tier inverted index over one channel of one country's targets."""

    _ARRAYS = ("ids", "offsets", "toks", "is_gen", "is_common", "w",
               "p_data", "p_indices", "p_indptr")

    def __init__(self, codes=None, offsets=None, toks=None, n_vocab=0,
                 df_cap=200_000, gen_df_cap=None):
        if codes is None:  # filled by `load`
            return
        self.ids, self.offsets, self.toks = codes, offsets, toks
        n_rec = len(codes)
        rec = np.repeat(np.arange(n_rec, dtype=np.int32), np.diff(offsets))
        df = np.bincount(toks, minlength=n_vocab)
        idf = np.log1p(n_rec / np.maximum(df, 1))
        gen_df_cap = df_cap if gen_df_cap is None else min(gen_df_cap, df_cap)
        indexed = (df > 0) & (df <= df_cap)
        self.is_gen = indexed & (df <= gen_df_cap)
        self.is_common = indexed & ~self.is_gen
        self.w = np.where(indexed, SHARED_UNIT + idf ** 2, 0.0)
        self.n_indexed_tokens = int(indexed.sum())
        self.n_gen_tokens = int(self.is_gen.sum())
        g = self.is_gen[toks]
        p = sparse.csr_matrix(
            (np.ones(int(g.sum()), np.float64), (toks[g], rec[g])),
            shape=(n_vocab, n_rec))
        self.p_data, self.p_indices = p.data, p.indices.astype(np.int32)
        self.p_indptr = p.indptr.astype(np.int32)
        self._finish()

    def _finish(self):
        self.postings = sparse.csr_matrix(
            (self.p_data, self.p_indices, self.p_indptr),
            shape=(len(self.p_indptr) - 1, len(self.ids)), copy=False)
        # scratch flags: set for one query's tokens, then reset. An O(n) lookup
        # instead of np.isin's sort, which dominated query time.
        self._flag = np.zeros(len(self.w), bool)

    def dump(self, path):
        """Save for memory-mapped sharing between query worker processes."""
        os.makedirs(path, exist_ok=True)
        for a in self._ARRAYS:
            np.save(os.path.join(path, a + ".npy"), getattr(self, a))

    @classmethod
    def load(cls, path):
        self = cls()
        for a in cls._ARRAYS:
            setattr(self, a, np.load(os.path.join(path, a + ".npy"), mmap_mode="r"))
        self._finish()
        return self

    def score(self, cand, qtoks):
        """Packed (SHARED_UNIT * shared + idf score) of each candidate vs qtoks."""
        pos, t = _gather(self.offsets, self.toks, cand)
        self._flag[qtoks] = True
        hit = self._flag[t]
        self._flag[qtoks] = False
        return np.bincount(pos[hit], weights=self.w[t[hit]], minlength=len(cand))

    def query_batch(self, offsets, toks, topk=30, min_shared=2,
                    rare_df_weight=6.0):
        """Yield (record_idx, packed_value) arrays, top-K by score, per query."""
        n_q = len(offsets) - 1
        wq = np.where(self.is_gen[toks], self.w[toks], 0.0)
        q = sparse.csr_matrix((wq, toks, offsets), shape=(n_q, self.postings.shape[0]))
        q.eliminate_zeros()
        r = (q @ self.postings).tocsr()
        for i in range(n_q):
            lo, hi = r.indptr[i], r.indptr[i + 1]
            if hi == lo:
                yield np.empty(0, np.int64), np.empty(0)
                continue
            cand = r.indices[lo:hi].astype(np.int64)
            v = r.data[lo:hi]
            qt = toks[offsets[i]:offsets[i + 1]]
            common = qt[self.is_common[qt]]
            if len(common):
                # Common tokens add at most `cmax` to any candidate, so one
                # whose rare-token score cannot reach the current K-th best is
                # dropped before the (costly) forward-index lookup. Exact.
                if len(cand) > topk:
                    sc0 = v - np.floor(v / SHARED_UNIT) * SHARED_UNIT
                    cmax = float((self.w[common] - SHARED_UNIT).sum())
                    kth = np.partition(sc0, len(sc0) - topk)[len(sc0) - topk]
                    alive = sc0 + cmax >= kth
                    cand, v = cand[alive], v[alive]
                v = v + self.score(cand, common)
            shared = np.floor(v / SHARED_UNIT)
            sc = v - shared * SHARED_UNIT
            keep = (shared >= min_shared) | (sc >= rare_df_weight)
            cand, v, sc = cand[keep], v[keep], sc[keep]
            if len(cand) > topk:
                sel = np.argpartition(-sc, topk)[:topk]
                cand, v, sc = cand[sel], v[sel], sc[sel]
            order = np.argsort(-sc, kind="stable")
            yield cand[order], v[order]


def _unpack(v):
    shared = np.floor(v / SHARED_UNIT)
    return (v - shared * SHARED_UNIT).astype(np.float32), shared.astype(np.int16)


def _batch_rows(off, toks, rows):
    lens = off[rows + 1] - off[rows]
    boff = np.r_[0, np.cumsum(lens)].astype(np.int64)
    btoks = np.concatenate([toks[off[r]:off[r + 1]] for r in rows]) \
        if len(rows) else np.empty(0, np.int32)
    return boff, btoks


COLS = ("s1", "cand", "score", "shared", "rank", "nscore", "nshared", "nrank",
        "cidx")


class TargetStats:
    """Streaming per-target contention: best / second-best `all` score and the
    number of Source-1 entities that retrieved the target. Exact over *all*
    queries without ever holding every pair (150M+ on the full data) at once."""

    def __init__(self, n):
        self.best = np.zeros(n, np.float32)
        self.second = np.zeros(n, np.float32)
        self.claims = np.zeros(n, np.int32)

    def update(self, cidx, score):
        if len(cidx) == 0:
            return
        self.claims += np.bincount(cidx, minlength=len(self.claims)).astype(np.int32)
        order = np.lexsort((-score, cidx))
        t, s = cidx[order], score[order]
        first = np.r_[True, t[1:] != t[:-1]]
        tt = t[first]
        c1 = s[first]
        c2 = np.zeros(len(tt), np.float32)
        is2 = np.r_[False, (t[1:] == t[:-1]) & first[:-1]]
        c2[np.searchsorted(tt, t[is2])] = s[is2]
        top = -np.sort(-np.stack([self.best[tt], self.second[tt], c1, c2]), axis=0)
        self.best[tt], self.second[tt] = top[0], top[1]


def query_rows(idx, qcodes, qtok, k_ch, min_shared, batch=64):
    """Both channels + union for a block of queries -> dict of COLS arrays.

    ``qtok[ch]`` is (offsets, toks) for these queries, in ``qcodes`` order.
    """
    out = {k: [] for k in COLS}
    ids = idx["all"].ids
    for b in range(0, len(qcodes), batch):
        e = min(b + batch, len(qcodes))
        per_ch, toks_i = {}, {}
        for ch in CHANNELS:
            off, toks = qtok[ch]
            boff = off[b:e + 1] - off[b]
            btoks = toks[off[b]:off[e]]
            per_ch[ch] = list(idx[ch].query_batch(boff, btoks, k_ch[ch], min_shared))
            toks_i[ch] = [btoks[boff[i]:boff[i + 1]] for i in range(e - b)]
        for i in range(e - b):
            ca, _ = per_ch["all"][i]
            cn, _ = per_ch["name"][i]
            union = np.union1d(ca, cn)
            if len(union) == 0:
                continue
            # every pair gets both channels' scores, whichever found it
            sc, sh = _unpack(idx["all"].score(union, toks_i["all"][i]))
            nsc, nsh = _unpack(idx["name"].score(union, toks_i["name"][i]))
            order = np.lexsort((-nsc, -sc))
            union, sc, sh, nsc, nsh = (union[order], sc[order], sh[order],
                                       nsc[order], nsh[order])
            nrank = np.full(len(union), NO_RANK, np.int16)
            srt = np.argsort(union)
            nrank[srt[np.searchsorted(union, cn, sorter=srt)]] = \
                np.arange(len(cn), dtype=np.int16)
            k = len(union)
            out["s1"].append(np.full(k, qcodes[b + i], np.int64))
            out["cand"].append(np.asarray(ids[union]))
            out["score"].append(sc)
            out["shared"].append(sh)
            out["rank"].append(np.arange(k, dtype=np.int16))
            out["nscore"].append(nsc)
            out["nshared"].append(nsh)
            out["nrank"].append(nrank)
            out["cidx"].append(union.astype(np.int32))
    empty = {"s1": np.int64, "cand": np.int64, "score": np.float32,
             "shared": np.int16, "rank": np.int16, "nscore": np.float32,
             "nshared": np.int16, "nrank": np.int16, "cidx": np.int32}
    return {k: (np.concatenate(v) if v else np.empty(0, empty[k]))
            for k, v in out.items()}


_QW: dict = {}


def _init_query_worker(path, k_ch, min_shared):
    _QW["idx"] = {ch: TargetIndex.load(os.path.join(path, ch)) for ch in CHANNELS}
    _QW["k_ch"], _QW["min_shared"] = k_ch, min_shared


def _query_task(task):
    qcodes, qtok = task
    return query_rows(_QW["idx"], qcodes, qtok, _QW["k_ch"], _QW["min_shared"])


def _tasks(qry, country, qcodes, sel, size):
    for b in range(0, len(sel), size):
        rows = sel[b:b + size]
        yield qcodes[rows], {ch: _batch_rows(*qry[(country, ch)][1:], rows)
                             for ch in CHANNELS}


def run(data_dir, prefix, out_path, topk=50, df_cap=200_000, min_shared=2,
        s1_sample=None, seed=0, aliases=None, batch=64, write_tsv=True,
        gen_df_cap=None, cache=None, name_topk=25, name_gen_df_cap=None,
        workers=1, store_sample=None, part_pairs=2_000_000):
    t0 = time.time()
    if cache and os.path.exists(cache):
        tgt, qry, n_vocab = load_cache(cache)
        print(f"[blocking] loaded token cache {cache} ({time.time() - t0:.0f}s)")
    else:
        amap = ertext.load_aliases(aliases)
        if amap:
            print(f"[blocking] alias lexicon: {len(amap)} entries")
        tgt, qry, n_vocab = tokenize_all(data_dir, prefix, amap, t0)
        if cache:
            save_cache(cache, tgt, qry, n_vocab)
            print(f"[blocking] saved token cache {cache}")
    countries = sorted({c for c, _ in qry})
    print(f"[blocking] vocabulary {n_vocab}; countries {countries}; "
          f"ready in {time.time() - t0:.0f}s")

    s1_order = [r for r, _, _, _ in iter_source(
        os.path.join(data_dir, f"{prefix}_source1.tsv"))]
    # Development mode: query only a random subset of Source-1 while keeping the
    # FULL Source-2/3 pool in the index. Shrinking the target pool instead (as a
    # naive subset would) makes blocking look far easier than it is.
    # NOTE: contention features (target_n_claims, rev_rank) need *every*
    # Source-1 record queried; for a faithful dev run block everything and
    # sample at `pipeline.py build --max-s1` instead.
    def sample(n):
        rng = np.random.default_rng(seed)
        return np.array(sorted(encode_id(x) for x in rng.choice(
            np.array(s1_order, dtype=object), size=min(n, len(s1_order)),
            replace=False)), np.int64)

    sampled = stored = None
    if s1_sample:
        sampled = sample(s1_sample)
        print(f"[blocking] querying a sample of {len(sampled)}/{len(s1_order)} "
              f"source-1 records against the full target pool (contention "
              f"features will NOT be faithful)")
    elif store_sample:
        stored = sample(store_sample)
        print(f"[blocking] querying all {len(s1_order)} source-1 records, "
              f"storing pairs for {len(stored)} of them")

    sidecar = scores_dir(out_path)
    if os.path.isdir(sidecar):
        shutil.rmtree(sidecar)
    os.makedirs(sidecar)
    buf = {k: [] for k in COLS if k != "cidx"}
    n_buf = n_part = n_pairs = 0

    def flush():
        nonlocal n_buf, n_part
        if not n_buf:
            return
        np.savez(os.path.join(sidecar, f"part_{n_part:04d}.npz"),
                 **{k: np.concatenate(v) for k, v in buf.items()})
        for v in buf.values():
            v.clear()
        n_part += 1
        n_buf = 0

    queried = []
    t_codes, t_best, t_second, t_claims = [], [], [], []
    k_ch = {"all": topk, "name": name_topk}
    gen_ch = {"all": gen_df_cap, "name": name_gen_df_cap}
    tmp_root = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                            "_blocking_index")
    for country in countries:
        qcodes = qry[(country, "all")][0]
        sel = np.flatnonzero(np.isin(qcodes, sampled)) if sampled is not None \
            else np.arange(len(qcodes))
        queried.append(qcodes[sel])
        if (country, "all") not in tgt:
            print(f"[blocking] {country}: no targets")
            continue
        idx = {}
        for ch in CHANNELS:
            idx[ch] = TargetIndex(*tgt.pop((country, ch)), n_vocab[ch], df_cap,
                                  gen_ch[ch])
            print(f"[blocking] {country}/{ch}: {len(idx[ch].ids)} targets, "
                  f"{idx[ch].n_indexed_tokens} indexed tokens "
                  f"({idx[ch].n_gen_tokens} generating), top-{k_ch[ch]}", flush=True)
        print(f"[blocking] {country}: {len(sel)} queries", flush=True)
        stats = TargetStats(len(idx["all"].ids))
        t_codes.append(np.array(idx["all"].ids))
        tq = time.time()
        task_size = batch * 8
        if workers > 1:
            # Index built once, shared by the workers through memory maps.
            path = os.path.join(tmp_root, country)
            for ch in CHANNELS:
                idx[ch].dump(os.path.join(path, ch))
            del idx
            pool = mp.get_context("spawn").Pool(
                workers, initializer=_init_query_worker,
                initargs=(path, k_ch, min_shared))
            results = pool.imap(_query_task,
                                _tasks(qry, country, qcodes, sel, task_size))
        else:
            pool = None
            results = (query_rows(idx, qc, qt, k_ch, min_shared, batch)
                       for qc, qt in _tasks(qry, country, qcodes, sel, task_size))
        done = 0
        for res in results:
            stats.update(res["cidx"], res["score"])
            if stored is not None:
                m = np.isin(res["s1"], stored)
                res = {k: v[m] for k, v in res.items()}
            for k in buf:
                buf[k].append(res[k])
            n_buf += len(res["s1"])
            n_pairs += len(res["s1"])
            if n_buf >= part_pairs:
                flush()
            done = min(done + task_size, len(sel))
            if done % (task_size * 100) < task_size or done == len(sel):
                el = time.time() - tq
                print(f"[blocking] {country}: {done}/{len(sel)} queries "
                      f"({el:.0f}s, {el / max(done, 1) * 1000:.2f} ms/query "
                      f"wall)", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            del idx
        t_best.append(stats.best)
        t_second.append(stats.second)
        t_claims.append(stats.claims)
        del stats
    flush()

    queried = np.concatenate(queried) if queried else np.empty(0, np.int64)
    tc = np.concatenate(t_codes) if t_codes else np.empty(0, np.int64)
    order = np.argsort(tc)
    np.savez(os.path.join(sidecar, "meta.npz"),
             queried=queried,
             stored=queried if stored is None else np.intersect1d(queried, stored),
             t_codes=tc[order],
             t_best=np.concatenate(t_best)[order] if t_best else np.empty(0, np.float32),
             t_second=np.concatenate(t_second)[order] if t_second else np.empty(0, np.float32),
             t_claims=np.concatenate(t_claims)[order] if t_claims else np.empty(0, np.int32))

    if write_tsv:
        results = {}
        for p in score_parts(sidecar):
            d = np.load(p)
            s1c, cc = d["s1"], d["cand"]
            starts = np.flatnonzero(np.r_[True, s1c[1:] != s1c[:-1]]) if len(s1c) \
                else np.zeros(0, np.int64)
            for lo, hi in zip(starts, np.r_[starts[1:], len(s1c)]):
                results[int(s1c[lo])] = cc[lo:hi]
        keep = sampled if sampled is not None else stored
        write_candidate_tsv(out_path, s1_order, results,
                            None if keep is None else {decode_id(c) for c in keep})

    print(f"[blocking] {n_pairs} candidate pairs stored "
          f"({n_pairs / max(len(queried) if stored is None else len(stored), 1):.1f} "
          f"per stored entity) in {n_part} part(s) -> {sidecar}; "
          f"total {time.time() - t0:.0f}s")


def write_candidate_tsv(path, s1_order, cands, restrict=None):
    """Submission-format candidate list: every Source-1 id gets one row."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for rid in s1_order:
            if restrict is not None and rid not in restrict:
                continue
            c = cands.get(encode_id(rid))
            ids = ",".join(decode_id(x) for x in c) if c is not None else ""
            f.write(f"{rid}\t{ids}\n")


def scores_dir(path):
    """'x/candidate_pairs.tsv' (or '..._scores', '..._scores.npz') -> sidecar dir."""
    if path.endswith(".npz"):
        path = path[:-4]
    if path.endswith("_scores"):
        return path
    return (path[:-4] if path.endswith(".tsv") else path) + "_scores"


def score_parts(path):
    d = scores_dir(path)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("part_"))


def load_meta(path):
    d = np.load(os.path.join(scores_dir(path), "meta.npz"))
    return {k: d[k] for k in d.files}


def load_scores(path):
    """Whole sidecar (all parts concatenated + meta) - for dev-sized runs only."""
    cols = {}
    for p in score_parts(path):
        d = np.load(p)
        for k in d.files:
            cols.setdefault(k, []).append(d[k])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out.update(load_meta(path))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50, help="K of the `all` channel")
    ap.add_argument("--name-topk", type=int, default=25,
                    help="K of the `name` channel (0 disables it)")
    ap.add_argument("--df-cap", type=int, default=200_000,
                    help="tokens above this df are ignored entirely")
    ap.add_argument("--gen-df-cap", type=int,
                    help="`all` channel: only tokens at or below this df "
                         "generate candidates; tokens up to --df-cap still add "
                         "score (default: same as --df-cap, a single tier)")
    ap.add_argument("--name-gen-df-cap", type=int,
                    help="same, for the `name` channel")
    ap.add_argument("--min-shared", type=int, default=2)
    ap.add_argument("--s1-sample", type=int,
                    help="query only N random source-1 records (dev mode); the "
                         "full source-2/3 pool is still indexed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aliases", help="alias lexicon TSV from aliases.py")
    ap.add_argument("--batch", type=int, default=64,
                    help="queries per sparse product (memory ~ batch x postings)")
    ap.add_argument("--cache", help="token-id cache (.npz): written if missing, "
                                    "reused if present. Delete it after changing "
                                    "aliases or tokenisation.")
    ap.add_argument("--no-tsv", action="store_true",
                    help="skip the submission-format TSV (large dev runs)")
    ap.add_argument("--workers", type=int, default=1,
                    help="query processes; they share the index via memory "
                         "maps written next to --out")
    ap.add_argument("--store-sample", type=int,
                    help="query EVERY source-1 record (faithful contention "
                         "statistics) but store pairs for only N random ones: "
                         "the realistic dev run")
    ap.add_argument("--part-pairs", type=int, default=2_000_000,
                    help="pairs per sidecar part file")
    a = ap.parse_args()
    run(a.data_dir, a.prefix, a.out, a.topk, a.df_cap, a.min_shared,
        a.s1_sample, a.seed, a.aliases, a.batch, not a.no_tsv, a.gen_df_cap,
        a.cache, a.name_topk, a.name_gen_df_cap, a.workers, a.store_sample,
        a.part_pairs)
