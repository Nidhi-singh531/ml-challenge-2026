#!/usr/bin/env bash
# Test-set tokenisation + blocking on an EC2 box (m7i-flex.large: 2 vCPU, 8 GB).
# Upload aws_bundle.tgz and this script to the home directory, then (inside tmux):
#     bash run_test_blocking.sh
# Result to download: ~/cp_all_scores.tgz  (the blocking sidecar directory + log)
set -euo pipefail
cd ~
sudo dnf install -y tmux tar >/dev/null 2>&1 || true
# any Python >= 3.9 works; prefer the newest available
PY=""
for v in python3.12 python3.11 python3; do
  if command -v "$v" >/dev/null 2>&1; then PY="$v"; break; fi
  sudo dnf install -y "$v" >/dev/null 2>&1 && command -v "$v" >/dev/null 2>&1 && { PY="$v"; break; } || true
done
echo "using $($PY --version)"
mkdir -p ml && cd ml
tar -xzf ~/aws_bundle.tgz
$PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy
s=code/business_entity_resolution/src
mkdir -p output work
# SAME settings as the laptop training run - never change one without the other
python $s/blocking.py --data-dir dataset/test --prefix test \
  --out output/cp_all.tsv --no-tsv --aliases work/aliases_full.tsv \
  --topk 50 --name-topk 25 --gen-df-cap 20000 --name-gen-df-cap 20000 \
  --cache work/tokcache_test.npz --workers 2 2>&1 | tee output/blocking_test.log
cd output && tar -czf ~/cp_all_scores.tgz cp_all_scores blocking_test.log
echo "DONE -> ~/cp_all_scores.tgz"
