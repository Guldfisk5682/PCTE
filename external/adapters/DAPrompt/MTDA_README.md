# Minimal RN50 MTDA adapter

Train one model on one labeled source and three unlabeled targets simultaneously.
Original upstream files are unchanged. The adapter uses the shared controlled
experiment harness in the sibling `spl` checkout.

```bash
python train_mtda.py --source Art --data-root /home/tangxunchi/office_home \
  --clip-checkpoint /home/tangxunchi/.cache/clip/RN50.pt \
  --seed 42 --steps 1000 --batch-size 30 --method DAPrompt
```

Use absolute `--output-dir` and `--cache-dir` paths when running outside SPL.
Defaults are relative to the current working directory. Do not use target
labels for training, threshold tuning, scheduling or checkpoint selection.

Full provenance, native losses, implementation repairs, budget differences and
comparison limitations: [/home/tangxunchi/workspace/spl/docs/vlm_competitors_protocol.md](/home/tangxunchi/workspace/spl/docs/vlm_competitors_protocol.md).
