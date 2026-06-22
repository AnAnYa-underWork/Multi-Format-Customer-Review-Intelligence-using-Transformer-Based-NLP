def chunk_text(text, chunk_size=200, overlap=40):
    words = text.split()

    if len(words) <= 300:
        return [text]  # no chunking

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += (chunk_size - overlap)

    return chunks

def process_chunks(documents):
    chunked_data = []

    for doc in documents:
        text = doc.get("text", "")
        source = doc.get("source", "unknown")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunked_data.append({
                "source": source,
                "chunk_id": i,
                "text": chunk,
                "rating": doc.get("rating", None),
                "date": doc.get("date", None),
                "product": doc.get("product", None)
            })

    return chunked_data