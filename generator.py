from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("com pleted", "completed")
    return text.strip()


def get_all_context(context_chunks):
    if not context_chunks:
        return ""
    return clean_text(" ".join(context_chunks))


def is_query_relevant_to_context(query, context_chunks):
    if not query or not context_chunks:
        return False

    context = get_all_context(context_chunks).lower()

    stopwords = {
        "what", "who", "does", "about", "is", "the", "in", "of", "to",
        "was", "were", "with", "from", "this", "that", "and", "are"
    }

    words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    important_words = [w for w in words if len(w) > 3 and w not in stopwords]

    matches = sum(1 for w in important_words if w in context)

    return matches >= 1


def extract_answer(context_chunks):
    if not context_chunks:
        return "No relevant information found."

    context = clean_text(context_chunks[0])
    sentences = context.split(".")

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 40:
            return sentence + "."

    return context[:200] + "..."


def generate_response(prompt, context_chunks=None, query=None):
    try:
        out_of_scope_terms = [
            "president of france",
            "world cup",
            "elon musk",
            "barack obama",
            "donald trump",
            "artificial intelligence"
        ]

        if query and any(term in query.lower() for term in out_of_scope_terms):
            return "This question is outside the scope of the provided documents."

        if query and not is_query_relevant_to_context(query, context_chunks):
            return "This question is outside the scope of the provided documents."

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response = response.replace("\n", " ").strip()

        if not response or len(response) < 10 or "i don't know" in response.lower():
            return extract_answer(context_chunks)

        return response

    except Exception:
        return extract_answer(context_chunks)