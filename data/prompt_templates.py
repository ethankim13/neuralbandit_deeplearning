import argparse

TEMPLATES = {
    "math":(
        "Solve the following math problems step by step. "
        "At the end, write the final answer as a number with no units, "
        'on its own line, in the form "#### <number>".\n\n'
        "Problem: {prompt}"
    ),
    "coding": (
        "Write the Python function described below. "
        "Output only code: the complete function, including its signature and any imports it needs, "
        "in a single ```python code block, with no explanation.\n\n"
        "{prompt}"
    ),
    "qa": (
        "Answer the following question in a few words. "
        "Give only the answer, with no explanation.\n\n"
        "Question: {prompt}"
    ),
}

# decoding settings, identical for every model
TEMPERATURE = 0.0                                       # greedy: same input -> same output
MAX_NEW_TOKENS = {"math": 512, "coding": 512, "qa": 64} # output cap per domain

def build_messages(domain, prompt):
    """Chat messages for one prompt. Pass these to the model's chat template later on"""
    return [{"role": "user", "content": TEMPLATES[domain].format(prompt = prompt)}]

def main():
    import pandas as pd
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default = "data/prompts.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.prompts)
    assert set(df["domain"]) <= set(TEMPLATES), f"no template for {set(df['domain']) - set(TEMPLATES)}"
    for d, p in zip(df["domain"], df["prompt"]):    # every row must format cleanly
        build_messages(d, p)

    source = df["id"].str.split("-").str[0]
    examples = df.groupby(source).head(1)
    for row in examples.itertuples():
        print(f"===== {row.id} (domain = {row.domain}, max_new_tokens = {MAX_NEW_TOKENS[row.domain]})")
        print(build_messages(row.domain, row.prompt)[0]["content"])
        print()
    print(f"All {len(df)} prompts formatted OK. temperature = {TEMPERATURE}")



if __name__ == "__main__":
    main()