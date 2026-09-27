"""Copy the rows of one country label from every source into a new data dir.

Used to re-block a single partition (blocking is partitioned by country, so the
other partitions' candidates are unaffected) and to curate bootstrap aliases.

    python country_subset.py --data-dir dataset/test --prefix test --country France --out dataset/test_fr
    python country_subset.py curate --base work/aliases_full.tsv --boot work/aliases_fr.tsv --out work/aliases_fr_curated.tsv
"""
import argparse
import os
import re


def subset(data_dir, prefix, country, out):
    os.makedirs(out, exist_ok=True)
    for i in (1, 2, 3):
        n = 0
        with open(os.path.join(data_dir, f"{prefix}_source{i}.tsv"), encoding="utf-8") as f, \
                open(os.path.join(out, f"{prefix}_source{i}.tsv"), "w", encoding="utf-8") as g:
            g.write(next(f))
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    g.write(line)
                    n += 1
        print(f"[subset] source{i}: {n} {country} rows -> {out}")


def curate(base, boot, out):
    """Keep bootstrap variants that look like abbreviations or typos of their word.

    Drops variants holding digits (house numbers such as '2bis' carry conflict
    evidence), single-letter canonicals, and pairs whose first letters differ
    ('pas' -> 'calais' is co-occurrence inside one place name, not an alias).
    """
    known = set()
    with open(base, encoding="utf-8") as f:
        next(f)
        known = {line.split("\t")[0] for line in f}
    kept = dropped = 0
    with open(boot, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        g.write(next(f))
        for line in f:
            x, y = line.split("\t")[:2]
            if x not in known and (re.search(r"\d", x) or len(y) < 2 or x[0] != y[0]):
                dropped += 1
                continue
            kept += x not in known
            g.write(line)
    print(f"[curate] kept {kept} new variants, dropped {dropped} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="subset", choices=["subset", "curate"])
    ap.add_argument("--data-dir")
    ap.add_argument("--prefix", default="test")
    ap.add_argument("--country")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base")
    ap.add_argument("--boot")
    a = ap.parse_args()
    if a.cmd == "curate":
        curate(a.base, a.boot, a.out)
    else:
        subset(a.data_dir, a.prefix, a.country, a.out)
