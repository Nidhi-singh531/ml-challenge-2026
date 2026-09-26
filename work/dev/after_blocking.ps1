$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$s = "code/business_entity_resolution/src"
while (-not ((Get-Content work/dev/blocking.log -Raw -ErrorAction SilentlyContinue) -match "total \d+s")) {
    if (-not (Get-Process -Id 34788 -ErrorAction SilentlyContinue)) { "blocking process exited without finishing"; exit 1 }
    Start-Sleep 60
}
& $py $s/analyze.py blocking --cand-scores work/dev/candidate_pairs_scores `
    --ground-truth dataset/train/train_ground_truth.tsv --data-dir dataset/train *> work/dev/analyze_blocking.log
& $py $s/pipeline.py build --data-dir dataset/train --prefix train `
    --cand-scores work/dev/candidate_pairs_scores `
    --ground-truth dataset/train/train_ground_truth.tsv `
    --aliases work/aliases_full.tsv --out work/dev/pairs40k.npz --max-s1 40000 --workers 5 *> work/dev/build40k.log
& $py $s/pipeline.py train --pairs work/dev/pairs40k.npz `
    --ground-truth dataset/train/train_ground_truth.tsv --out work/dev/model40k.pkl *> work/dev/train40k.log
& $py $s/analyze.py errors --pairs work/dev/pairs40k.npz --model work/dev/model40k.pkl `
    --ground-truth dataset/train/train_ground_truth.tsv --data-dir dataset/train *> work/dev/errors40k.log
"done"
