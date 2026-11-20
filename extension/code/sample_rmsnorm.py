"""Fixed-prompt continuations from an RMSNorm checkpoint. Same defaults as src.sample."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from extension.code.rmsnorm_gpt import load_rmsnorm_checkpoint  # noqa: E402
from src.sample import PROMPTS  # noqa: E402
from src.utils import sha256, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_rmsnorm_checkpoint(args.checkpoint, args.device)
    if sha256(args.tokenizer) != checkpoint["tokenizer_sha256"]:
        raise ValueError("checkpoint/tokenizer mismatch")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    samples = []
    for number, prompt in enumerate(PROMPTS):
        ids = tokenizer.encode(prompt).ids or [tokenizer.token_to_id("<|endoftext|>")]
        rng = torch.Generator(device=args.device).manual_seed(args.seed + number)
        x = torch.tensor([ids], device=args.device)
        y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k,
                           eos_id=tokenizer.token_to_id("<|endoftext|>"), generator=rng)
        samples.append({"prompt": prompt, "continuation": tokenizer.decode(y[0, len(ids):].tolist()),
                        "full_text": tokenizer.decode(y[0].tolist())})
    write_json(args.out, {"checkpoint_sha256": sha256(args.checkpoint), "seed": args.seed,
        "temperature": args.temperature, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
        "architecture": "rmsnorm", "samples": samples})
    print(json.dumps(samples, indent=2), flush=True)


if __name__ == "__main__":
    main()
