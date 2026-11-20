# Extension: RMSNorm vs LayerNorm (3036807088)

## Question
With the baseline GPT fixed (6 layers, 6 heads, width 384, tied embeddings, AdamW, seed 1337), does replacing every LayerNorm with RMSNorm change next-token validation loss on the fixed TinyStories subset?

Hypothesis: RMSNorm removes mean-centering and the bias, which can stabilize the residual stream and slightly change optimization speed. A gain is not assumed. The comparison is one factor only.

## What changed
- New module `RMSNorm` (scale only, fp32 variance, eps 1e-5). Citation: Zhang and Sennrich, 2019, Root Mean Square Layer Normalization.
- `RMSBlock` and final `ln_f` use RMSNorm. Attention, MLP, residuals, embeddings, and the loss are the completed core implementation.
- Optimizer, YAML hyperparameters, vocabulary 8192, context 256, and the 10.24M-token budget match `configs/baseline.yaml`.
- This code is not used for the core `best.pt`. Core runs stay on LayerNorm.

RMSNorm has a weight and no bias, so the extension model has fewer parameters than the 13,891,584 baseline (13 LayerNorms x 384 bias terms).

## File map
- `code/rmsnorm_gpt.py` — model and checkpoint loader (`architecture: rmsnorm`)
- `code/train_rmsnorm.py` — official training loop with the RMSNorm model
- `code/evaluate_rmsnorm.py` — full-split FP32 evaluation
- `code/sample_rmsnorm.py` — the three supplied prompts
- `configs/rmsnorm.yaml` — baseline hyperparameters
- `results/` — put the run evidence here before upload

## Commands
From the starter root, after `data/` is in place:

```bash
python extension/code/train_rmsnorm.py \
  --config extension/configs/rmsnorm.yaml \
  --out_dir outputs/exp_rmsnorm --device cuda

python extension/code/evaluate_rmsnorm.py \
  --checkpoint outputs/exp_rmsnorm/best.pt \
  --tokens data/val.bin --max-tokens 0 \
  --out outputs/exp_rmsnorm/evaluation.json

python extension/code/sample_rmsnorm.py \
  --checkpoint outputs/exp_rmsnorm/best.pt \
  --out outputs/exp_rmsnorm/samples.json
```

Copy `metrics.json`, `train.csv`, `environment.json`, `evaluation.json`, and `samples.json` into `extension/results/`. Plot the baseline and RMSNorm `eval.jsonl` curves into the extension report. Do not load this checkpoint with `src.evaluate` or `src.sample`; those builders are LayerNorm-only.

## Dependencies
The starter `requirements.txt` only. No extra packages. No pretrained weights.
