"""Write extension_report.pdf with the standard library. Re-run after the GPU job and edit NUMBERS."""
from pathlib import Path


LINES = [
    "Extension: RMSNorm instead of LayerNorm",
    "Student 3036807088. Separate from the two-page core report.",
    "",
    "Question. Does RMSNorm change next-token loss when every other",
    "factor matches the released baseline?",
    "",
    "Method. Zhang and Sennrich (2019): normalize by the root mean",
    "square and apply a learned scale. No mean subtraction and no bias.",
    "Variance is computed in fp32. eps = 1e-5, the LayerNorm default.",
    "Pre-norm residuals are unchanged: x = x + attn(norm(x)), then",
    "x = x + mlp(norm(x)). Attention, GELU MLP, tied embeddings,",
    "vocabulary 8192, context 256, AdamW, seed 1337, and the",
    "10,240,000-token budget are the baseline. The model is trained",
    "from random initialization on the supplied training split only.",
    "",
    "Control. The core baseline (LayerNorm), not the 4-layer run and",
    "not the learning-rate run. One factor changes.",
    "",
    "Evidence to fill after training. Parameter count, processed tokens,",
    "precision (trainer bf16; public eval fp32), hardware, runtime,",
    "full-split validation cross-entropy and perplexity for baseline",
    "and RMSNorm, plus the loss curves and the three fixed prompts",
    "(temperature 0.8, top-k 40, seed 2026). Do not compare trainer",
    "sampled validation with full-split float32 evaluation.",
    "",
    "How to read the result. A lower loss is not required. If RMSNorm",
    "is worse, the bias and mean statistics in LayerNorm may matter on",
    "this short-story subset, or 2500 steps may be too few to separate",
    "the norms. A single seed is not statistical significance.",
    "Limitation: the parameter count is slightly smaller because each",
    "RMSNorm drops a bias vector, so the comparison is not perfectly",
    "parameter-matched. That gap is reported, not hidden.",
]


def pdf_bytes(lines):
    def esc(text):
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    commands = ["BT", "/F1 11 Tf", "50 800 Td", "14 TL"]
    for line in lines:
        commands.append(f"({esc(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    stream = "\n".join(commands).encode()
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(sum(len(c) for c in chunks))
        chunks.append(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = sum(len(c) for c in chunks)
    chunks.append(f"xref\n0 {len(objects) + 1}\n".encode())
    chunks.append(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        chunks.append(f"{off:010d} 00000 n \n".encode())
    chunks.append(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return b"".join(chunks)


def main():
    path = Path(__file__).resolve().parents[1] / "extension_report.pdf"
    path.write_bytes(pdf_bytes(LINES))
    print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
