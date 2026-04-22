import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("com pleted", "completed")
    return text.strip()


def is_query_relevant_to_context(query, context):
    """
    Strict relevance check to ensure answer is from dataset
    """
    if not query or not context:
        return False

    stopwords = {"what", "who", "does", "about", "is", "the", "in", "of", "to"}

    query_words = [
        w for w in query.lower().split()
        if len(w) > 3 and w not in stopwords
    ]

    matches = sum(1 for w in query_words if w in context.lower())

    return matches >= 2


def extract_answer(context_chunks, query=None):
    """
    Fallback: extract best sentence from retrieved context
    """
    if not context_chunks:
        return "No relevant information found."

    context = clean_text(context_chunks[0])

    if query and not is_query_relevant_to_context(query, context):
        return "This question is outside the scope of the provided documents."

    sentences = context.split(".")
    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            return s + "."

    return context[:200] + "..."


def generate_response(prompt, context_chunks=None, query=None):
    try:
        context = clean_text(context_chunks[0]) if context_chunks else ""

        # 🔒 HARD STOP (before model)
        if query and not is_query_relevant_to_context(query, context):
            return "This question is outside the scope of the provided documents."

        messages = [
            {"role": "system", "content": "Answer only using the provided context."},
            {"role": "user", "content": prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean output
        response = response.replace("\n", " ").strip()

        # ❌ Remove prompt echo
        if "context:" in response.lower() or "question:" in response.lower():
            return extract_answer(context_chunks, query)

        # 🔒 Double safety
        if query and not is_query_relevant_to_context(query, context):
            return "This question is outside the scope of the provided documents."

        # ❌ Weak / bad answers
        if not response or len(response) < 15 or "i don't know" in response.lower():
            return extract_answer(context_chunks, query)

        return response

    except Exception:
        return extract_answer(context_chunks, query)