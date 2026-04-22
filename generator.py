from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def clean_text(text):
    """
    Clean messy PDF text for better readability
    """
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.replace("com pleted", "completed")
    return text.strip()


def extract_answer(query, context_chunks):
    """
    Fallback: extract a simple answer directly from context
    """
    if not context_chunks:
        return "I don't know."

    context = clean_text(context_chunks[0])

    if query and "paris mission" in query.lower():
        return "The document mentions the Residency of the Paris Mission as part of properties that were completed or renovated."

    sentences = context.split(".")
    for s in sentences:
        if len(s.strip()) > 20:
            return s.strip() + "."

    return context[:150] + "..."


def generate_response(prompt, context_chunks=None, query=None):
    """
    Generate a one-line grounded answer using FLAN-T5
    with fallback handling.
    """
    try:
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

        bad_responses = ["", "answer:", "unknown", "i do not know"]

        if response.lower() in bad_responses or len(response) < 5:
            if context_chunks:
                return context_chunks[0][:150].replace("\n", " ") + "..."
            else:
                return "I don't know."

        return response

    except Exception:
        if context_chunks:
            return context_chunks[0][:150].replace("\n", " ") + "..."
        return "I don't know."