
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

def load_model():
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


def generate_response(prompt, tokenizer, model):
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    generated = outputs[0][inputs.input_ids.shape[1]:]

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer, model = load_model()

    question = "What is 12 multiplied by 8?"
    answer = generate_response(question, tokenizer, model)

    print("Model response:", answer)