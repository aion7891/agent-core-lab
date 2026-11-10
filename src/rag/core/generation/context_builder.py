def build_context(chunks):
    """Join chunks into one text block the LLM can read and cite."""
    passages = []

    for chunk in chunks:
        passages.append(
            f"[{chunk['id']}] Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(passages)
