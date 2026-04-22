def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
Answer the question in ONE short sentence using ONLY the context below.

If the answer is partially present, summarize what is available.

If the answer is not in the context, say: "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt