def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
Context:
{context}

Question:
{query}

Answer in one short sentence:
"""
    return prompt