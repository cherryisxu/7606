"""FP32 full-split evaluation for an RMSNorm checkpoint.

Official src.evaluate loads the core LayerNorm GPT and cannot load these weights.
Protocol matches src.evaluate.evaluate: non-overlapping windows, float32, max-tokens 0.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from extension.code.rmsnorm_gpt import load_rmsnorm_checkpoint  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.utils import sha256, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_rmsnorm_checkpoint(args.checkpoint, args.device)
    if sha256(args.tokenizer) != checkpoint["tokenizer_sha256"]:
        raise ValueError("checkpoint/tokenizer mismatch")
    tokens = np.memmap(args.tokens, dtype="<u2", mode="r")
    result = evaluate(model, tokens, args.batch_size, args.max_tokens, args.device)
    result.update(checkpoint_sha256=sha256(args.checkpoint), tokens_sha256=sha256(args.tokens),
                  model_parameters=model.parameter_count(), checkpoint_step=checkpoint["step"],
                  architecture="rmsnorm")
    write_json(args.out, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
