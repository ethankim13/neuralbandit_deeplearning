import argparse
import json
from pathlib import Path
import pandas as pd

def load_hf(path, config, split):
    from datasets import load_dataset
    
    return load_dataset(path, config, split = split).to_pandas()

LOADERS = {
    # our three domain models
    "gsm8k": lambda: load_hf("openai/gsm8k", "main", "test"),
    "humaneval": lambda: load_hf("openai/openai_humaneval", None, "test"),
    "mbpp": lambda: load_hf("google-research-datasets/mbpp", "full", "test"),
    "triviaqa": lambda: load_hf("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
}

# domain specific converters (raw HF columns -> id, domain, prompt, reference)
def gsm8k_rows(df):
    answers = df["answer"].str.split("####").str[-1].str.strip().str.replace(",", "")

    return pd.DataFrame({
        "id":[f"gsm8k-{i:04d}" for i in range(len(df))],
        "domain": "math",
        "prompt": df["question"].str.strip(),
        "reference": [json.dumps({"answer": a}) for a in answers],
    })

def humaneval_rows(df):
    tests = [f"{t}\n\ncheck({ep})\n" for t, ep in zip(df["test"], df["entry_point"])]

    return pd.DataFrame({
        "id": "humaneval-" + df["task_id"].str.split("/").str[1],
        "domain": "coding",
        "prompt": df["prompt"],
        "reference": [json.dumps({"tests": t}) for t in tests],
    })

def mbpp_rows(df):
    prompts, refs = [], []
    for text, setup, tests in zip(df["text"], df["test_setup_code"], df["test_list"]):
        tests = list(tests)
        prompts.append(f"{text.strip()}\nYour code should pass this test:\n{tests[0]}")
        refs.append(json.dumps({"tests": "\n".join([setup.strip(), *tests]).strip() + "\n"}))

    return pd.DataFrame({
        "id": [f"mbpp-{t}" for t in df["task_id"]],
        "domain": "coding",
        "prompt": prompts,
        "reference": refs,
    })

def triviaqa_rows(df):
    df = df.drop_duplicates("question_id")
    refs = [json.dumps({
        "value": str(a["value"]),
        "aliases": [str(x) for x in a ["aliases"]],
        "normalized_aliases": [str(x) for x in a["normalized_aliases"]],
    }) for a in df["answer"]]

    return pd.DataFrame({
        "id": "triviaqa-" + df["question_id"].astype(str),
        "domain": "qa",
        "prompt": df["question"].str.strip(),
        "reference": refs,
    })

# sampling
def dedupe(df, name):
    out = df.drop_duplicates("prompt")
    if len(out) < len(df):
        print(f"    {name}: dropped {len(df) - len(out)} duplicate prompts")
    return out.reset_index(drop = True)

def sample(df, n, seed):
    if len(df) <= n:
        return df
    return df.sample(n = n, random_state = seed).sort_index()

def build(n, seed):
    print("Loading datasets...")
    math = dedupe(gsm8k_rows(LOADERS["gsm8k"]()), "gsm8k")
    he = dedupe(humaneval_rows(LOADERS["humaneval"]()), "humaneval")
    mbpp = dedupe(mbpp_rows(LOADERS["mbpp"]()), "mbpp")
    qa = dedupe(triviaqa_rows(LOADERS["triviaqa"]()), "triviaqa")

    # for coding, keep all of HumanEval, fill the rest of the domain from MBPP
    he = sample(he, n, seed)
    mbpp = sample(mbpp, max(n - len(he), 0), seed)

    out = pd.concat([sample(math, n, seed), he, mbpp, sample(qa, n, seed)], ignore_index = True)
    return out[["id", "domain", "prompt", "reference"]]

def check(df):
    assert df["id"].is_unique, "duplicate ids"
    assert df["prompt"].str.strip().str.len().gt(0).all(), "empty prompt"
    for dom, ref in zip(df["domain"], df["reference"]):
        r = json.loads(ref)
        ok = {"math": lambda: r["answer"] != "",
            "coding": lambda: "assert" in r["tests"],
            "qa": lambda: len(r["normalized_aliases"]) > 0}[dom]()
        assert ok, f"bad reference for {dom}: {ref[:200]}"

def summarize(df):
    src = df["id"].str.split("-").str[0]
    print("\nrows per domain / source:")
    print(df.groupby([df["domain"], src]).size().to_string())
    print("\nPrompt length (chars) by domain:")
    print(df.groupby("domain")["prompt"].apply(lambda s: s.str.len().describe()[["min", "50%", "max"]])
        .unstack().astype(int).to_string())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-domain", type = int, default = 600)
    ap.add_argument("--seed", type = int, default = 0)
    ap.add_argument("--out", default = "data/prompts.parquet")
    args = ap.parse_args()

    df = build(args.n_per_domain, args.seed)
    check(df)
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(out, index = False)
    summarize(df)
    print(f"\nWrote {len(df)} rows -> {out}")

if __name__ == "__main__":
    main()