# extraction.py

import re

# =====================================================
# ONLY KEEP STRUCTURED EXTRACTION HERE
# =====================================================
# NLP CLASSIFICATION NOW MOVED TO:
# analyzer.py
#
# This file should ONLY handle:
# - product extraction
# - date extraction
# - rating extraction
# - confidence
# =====================================================

# -----------------------------
# PRODUCT EXTRACTION
# -----------------------------
def extract_product(text):

    text = text.lower()

    patterns = [

        # Product ID: XXXXX
        r'product\s*id[:\s]*([a-z0-9]+)',

        # for product XXXXX
        r'for\s+product\s+([a-z0-9]+)',

        # product XXXXX
        r'product\s+([a-z0-9]+)',

        # ID XXXXX after "product"
        r'product.*?([a-z0-9]{8,15})',

    ]

    for p in patterns:

        match = re.search(
            p,
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(1).upper()

    # --------------------------------
    # FALLBACK:
    # detect 9-12 digit product ids
    # --------------------------------

    match = re.search(
        r'\b\d{9,12}\b',
        text
    )

    if match:
        return match.group(0)

    return None

# -----------------------------
# DATE EXTRACTION
# -----------------------------
def extract_date(text):

    patterns = [

        # 05 21, 2014
        r'(\d{2}\s\d{2},\s\d{4})',

        # 05/21/2014
        r'(\d{2}/\d{2}/\d{4})',

        # 2014-05-21
        r'(\d{4}-\d{2}-\d{2})',

        # May 21, 2014
        r'(\w+\s\d{1,2},\s\d{4})'
    ]

    for p in patterns:

        match = re.search(
            p,
            text
        )

        if match:
            return match.group(1)

    return None

# -----------------------------
# RATING EXTRACTION
# -----------------------------
def extract_rating(text):

    patterns = [

        # 5.0 stars
        r'(\d+(\.\d+)?)\s*stars',

        # rated 4.0
        r'rated\s*(\d+(\.\d+)?)',

        # rating: 5
        r'rating[:\s]*(\d+(\.\d+)?)'
    ]

    for p in patterns:

        match = re.search(
            p,
            text,
            re.IGNORECASE
        )

        if match:

            return float(match.group(1))

    return None

# =====================================================
# MAIN EXTRACTION
# =====================================================
def extract_info(
        text,
        rating=None,
        product=None,
        date=None
):

    # --------------------------------
    # STRUCTURED EXTRACTION ONLY
    # --------------------------------

    product = product or extract_product(text)

    date = date or extract_date(text)

    rating = rating or extract_rating(text)

    return {

        "product": product,

        "date": date,

        "rating": rating
    }

# =====================================================
# PROCESS ALL DATA
# =====================================================
def process_extraction(chunked_data):

    extracted = []

    for item in chunked_data:

        result = extract_info(

            text=item["text"],

            rating=item.get("rating"),

            product=item.get("product"),

            date=item.get("date")
        )

        # --------------------------------
        # SOURCE
        # --------------------------------

        result["source"] = item["source"]

        # --------------------------------
        # CONFIDENCE
        # --------------------------------

        fields = [
            "product",
            "date",
            "rating"
        ]

        filled = sum(
            1 for f in fields
            if result.get(f) not in [None, ""]
        )

        result["confidence"] = round(
            filled / len(fields),
            2
        )

        extracted.append(result)

    return extracted