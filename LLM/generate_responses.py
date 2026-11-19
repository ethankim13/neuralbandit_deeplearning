
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "math": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "coding": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "general": "google/gemma-2-2b-it",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_parquet("data/prompts.parquet")
    df = df.head(args.limit)

    model_name = MODELS[args.model]
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    output_path = Path(f"data/responses_{args.model}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for _, row in df.iterrows():
            messages = [{"role": "user", "content": row["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                [text], return_tensors="pt"
            ).to(model.device)

            start = time.perf_counter()

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            generated = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(
                generated, skip_special_tokens=True
            )

            record = {
                "id": row["id"],
                "domain": row["domain"],
                "model": args.model,
                "response": response,
                "latency_seconds": elapsed
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

            print(f"Completed {row['id']} ({elapsed:.1f}s)")

    print(f"Saved responses to {output_path}")


if __name__ == "__main__":
    main()