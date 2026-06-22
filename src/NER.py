import spacy

nlp = spacy.load("en_core_web_sm")

def extract_context_ner(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]

    useful_sentences = []

    for sent in sentences:
        s = sent.lower().strip()

        # ❌ skip junk / headers
        if any(x in s for x in [
            "reviewer", "id", "product id", "report",
            "analysis", "summary", "helpful",
            "this report", "evaluation"
        ]):
            continue

        # ❌ skip filler sentences
        if any(x in s for x in [
            "so yeah", "i guess", "lol", "okay"
        ]):
            continue

        # ❌ skip too short
        if len(s) < 25:
            continue

        if any(x in s for x in [
            "this is a review", "review by", "for product"
        ]):
            continue

        # ✅ keep opinion/experience sentences
        if any(x in s for x in [
            "good", "great", "bad", "waste", "perfect",
            "broke", "works", "fit", "problem",
            "issue", "like", "love", "hate"
        ]):
            useful_sentences.append(sent)

    # ✅ if found good sentences
    if useful_sentences:
        return useful_sentences[0]

    # 🔁 smarter fallback
    for sent in sentences:
        s = sent.lower().strip()

        if len(s) > 30 and not any(x in s for x in [
            "review", "id", "product", "report"
        ]):
            return sent.strip()

    return text[:100]