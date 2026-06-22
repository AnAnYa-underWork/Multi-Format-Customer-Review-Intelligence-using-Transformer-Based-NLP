# ============================================
# app.py
# FINAL HYBRID NLP + RAG VERSION
# ============================================
import re

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# -----------------------------
# YOUR PIPELINE
# -----------------------------
from ingestion import load_all_data
from chunking import process_chunks
from extraction import process_extraction
from NER import extract_context_ner

# -----------------------------
# RAG PIPELINE
# -----------------------------
from analyzer import analyze_review
from vector_db import build_index
from retriever_RAG import retrieve_context

# -----------------------------
# GENERATION
# -----------------------------
from transformers import pipeline

generator = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Hybrid NLP-RAG Platform",
    layout="wide"
)

st.title(
    "Hybrid NLP-RAG Multi-Format Review Intelligence Platform"
)

st.write("""
Upload CSV, TXT, DOCX, and PDF files to:
- Extract structured information
- Generate analytics dashboards
- Perform semantic RAG-based querying
""")

# ============================================
# TEMP FOLDER
# ============================================

TEMP_FOLDER = "temp_data"

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# ============================================
# SESSION STATE
# ============================================

if "processed" not in st.session_state:
    st.session_state.processed = False

# ============================================
# FILE UPLOAD
# ============================================

uploaded_files = st.file_uploader(
    "Upload Files",
    type=["csv", "txt", "docx", "pdf"],
    accept_multiple_files=True
)

# ============================================
# FIELD SELECTION
# ============================================

st.subheader("Select Fields")

col1, col2, col3 = st.columns(3)

with col1:
    use_product = st.checkbox(
        "Product ID",
        value=True
    )

    use_date = st.checkbox(
        "Date",
        value=True
    )

with col2:
    use_rating = st.checkbox(
        "Rating",
        value=True
    )

    use_sentiment = st.checkbox(
        "Sentiment",
        value=True
    )

with col3:
    use_issue = st.checkbox(
        "Issue Type",
        value=True
    )

    use_context = st.checkbox(
        "Context",
        value=True
    )

selected_fields = []

if use_product:
    selected_fields.append("product")

if use_date:
    selected_fields.append("date")

if use_rating:
    selected_fields.append("rating")

if use_sentiment:
    selected_fields.append("sentiment")

if use_issue:
    selected_fields.append("issue_type")

if use_context:
    selected_fields.append("context")

# ============================================
# SAVE FILES
# ============================================

if uploaded_files:

    os.makedirs(
        os.path.join(TEMP_FOLDER, "complaints"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(TEMP_FOLDER, "summaries"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(TEMP_FOLDER, "reports"),
        exist_ok=True
    )

    for file in uploaded_files:

        filename = file.name.lower()

        # TXT
        if filename.endswith(".txt"):

            save_path = os.path.join(
                TEMP_FOLDER,
                "complaints",
                file.name
            )

        # DOCX
        elif filename.endswith(".docx"):

            save_path = os.path.join(
                TEMP_FOLDER,
                "summaries",
                file.name
            )

        # PDF
        elif filename.endswith(".pdf"):

            save_path = os.path.join(
                TEMP_FOLDER,
                "reports",
                file.name
            )

        # CSV
        elif filename.endswith(".csv"):

            save_path = os.path.join(
                TEMP_FOLDER,
                file.name
            )

        else:
            continue

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

# ============================================
# PROCESS BUTTON
# ============================================

if st.button("Process Files"):

    with st.spinner(
        "Running Hybrid NLP-RAG Pipeline..."
    ):

        try:

            # ====================================
            # LOAD DATA
            # ====================================

            data = load_all_data(TEMP_FOLDER)

            # ====================================
            # CHUNK DATA
            # ====================================

            chunked = process_chunks(data)

            # ====================================
            # STRUCTURED EXTRACTION
            # ====================================

            extracted = process_extraction(
                chunked
            )

            # ====================================
            # CONTEXT EXTRACTION
            # ====================================

            for i in range(len(extracted)):

                text = chunked[i]["text"]

                context = extract_context_ner(text)

                extracted[i]["context"] = context

            # ====================================
            # SEMANTIC ANALYSIS
            # ====================================

            for result in extracted:

                analysis = analyze_review(
                    result.get("context", ""),
                    result.get("context", "")
                )

                result.update(analysis)

            # ====================================
            # FINAL TABLE
            # ====================================

            final_rows = []

            for row in extracted:

                filtered = {}

                for field in selected_fields:
                    filtered[field] = row.get(field)

                filtered["keywords"] = row.get(
                    "keywords"
                )

                filtered["severity"] = row.get(
                    "severity"
                )

                filtered["source"] = row.get(
                    "source"
                )

                filtered["confidence"] = row.get(
                    "confidence"
                )

                final_rows.append(filtered)

            df = pd.DataFrame(final_rows)

            # ====================================
            # CREATE RAG DOCUMENTS
            # ====================================

            documents = []

            for i, row in enumerate(extracted):
                doc_text = f"""
                Product ID:
                {row.get('product')}

                Review:
                {row.get('context')}

                Issue:
                {row.get('issue_type')}

                Sentiment:
                {row.get('sentiment')}
                """

                documents.append({
                    "id": i,
                    "text": doc_text,
                    "source": row.get("source")
                })

            # ====================================
            # BUILD VECTOR DB
            # ====================================

            index, texts = build_index(documents)

            # ====================================
            # SAVE SESSION STATE
            # ====================================

            st.session_state["df"] = df
            st.session_state["index"] = index
            st.session_state["texts"] = texts
            st.session_state["extracted"] = extracted


            st.session_state.processed = True

            st.success(
                "Processing Complete!"
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

# ============================================
# SHOW DASHBOARD
# ============================================

if st.session_state.processed:

    df = st.session_state["df"]

    index = st.session_state["index"]

    texts = st.session_state["texts"]

    extracted = st.session_state["extracted"]

    tab1, tab2 = st.tabs([
        "Analytics Dashboard",
        "Semantic RAG Chat"
    ])

    # ========================================
    # ANALYTICS TAB
    # ========================================

    with tab1:

        st.subheader("Structured Output")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ====================================
        # DOWNLOAD CSV
        # ====================================

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="structured_output.csv",
            mime="text/csv"
        )

        # ====================================
        # ANALYTICS DASHBOARD
        # ====================================

        st.subheader(
            "Analytics Dashboard"
        )

        # ------------------------------------
        # ROW 1
        # ------------------------------------

        col_a, col_b = st.columns(2)

        # SENTIMENT
        if "sentiment" in df.columns:

            with col_a:

                sentiment_counts = (
                    df["sentiment"]
                    .value_counts()
                )

                fig, ax = plt.subplots(
                    figsize=(6, 4)
                )

                ax.bar(
                    sentiment_counts.index,
                    sentiment_counts.values,
                    color=[
                        "#00C2A8",
                        "#FF6B6B",
                        "#FFD93D"
                    ][:len(sentiment_counts)]
                )

                ax.set_title(
                    "Sentiment Overview"
                )

                st.pyplot(fig)

        # ISSUE
        if "issue_type" in df.columns:

            with col_b:

                issue_counts = (
                    df["issue_type"]
                    .value_counts()
                )

                fig, ax = plt.subplots(
                    figsize=(6, 4)
                )

                ax.bar(
                    issue_counts.index,
                    issue_counts.values,
                    color="#4D96FF"
                )

                plt.xticks(rotation=20)

                ax.set_title(
                    "Issue Categories"
                )

                st.pyplot(fig)

        # ------------------------------------
        # ROW 2
        # ------------------------------------

        col_c, col_d = st.columns(2)

        # RATING
        if "rating" in df.columns:

            with col_c:

                rating_series = pd.to_numeric(
                    df["rating"],
                    errors="coerce"
                ).dropna()

                if not rating_series.empty:

                    fig, ax = plt.subplots(
                        figsize=(6, 4)
                    )

                    ax.hist(
                        rating_series,
                        bins=5,
                        color="#9B5DE5"
                    )

                    ax.set_title(
                        "Rating Distribution"
                    )

                    st.pyplot(fig)

        # SEVERITY
        with col_d:

            severity_counts = (
                df["severity"]
                .value_counts()
            )

            fig, ax = plt.subplots(
                figsize=(6, 4)
            )

            ax.bar(
                severity_counts.index,
                severity_counts.values,
                color="#F15BB5"
            )

            ax.set_title(
                "Severity Levels"
            )

            st.pyplot(fig)

        # ------------------------------------
        # CONFIDENCE
        # ------------------------------------

        st.subheader(
            "Confidence Distribution"
        )

        conf_series = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        conf_counts = (
            conf_series
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.bar(
            conf_counts.index.astype(str),
            conf_counts.values,
            color="#06D6A0"
        )

        ax.set_title(
            "Confidence Scores"
        )

        st.pyplot(fig)

    # ========================================
    # RAG TAB
    # ========================================

    with tab2:

        st.subheader(
            "Semantic Question Answering"
        )

        # =================================
        # STORE QUERY IN SESSION STATE
        # =================================

        if "last_query" not in st.session_state:
            st.session_state.last_query = ""

        query = st.chat_input(
            "Ask questions about uploaded documents"
        )

        # ---------------------------------
        # SAVE QUERY
        # ---------------------------------

        if query:
            st.session_state.last_query = query

        # ---------------------------------
        # DISPLAY LAST QUERY
        # ---------------------------------

        if st.session_state.last_query:
            st.markdown(
                f"**Last Question:** "
                f"{st.session_state.last_query}"
            )

        if query:

            query_lower = query.lower()

            filtered_docs = []

            for t in texts:
                filtered_docs.append({
                    "text": t
                })

            # -----------------------------------
            # SMART METADATA FILTERING
            # -----------------------------------

            if "durability" in query_lower:

                filtered_docs = [
                    d for d in filtered_docs
                    if "durability_issue"
                       in d["text"].lower()
                ]

            elif "performance" in query_lower:

                filtered_docs = [
                    d for d in filtered_docs
                    if "performance_issue"
                       in d["text"].lower()
                ]

            elif "design" in query_lower:

                filtered_docs = [
                    d for d in filtered_docs
                    if "design_issue"
                       in d["text"].lower()
                ]

            elif "negative" in query_lower:

                filtered_docs = [
                    d for d in filtered_docs
                    if "negative"
                       in d["text"].lower()
                ]

            # -----------------------------------
            # FALLBACK
            # -----------------------------------

            if len(filtered_docs) == 0:

                filtered_docs = []

                for t in texts:
                    filtered_docs.append({
                        "text": t
                    })

            # -----------------------------------
            # RETRIEVE CONTEXT
            # -----------------------------------

            retrieved_docs = retrieve_context(
                query,
                filtered_docs,
                k=5
            )


            # =================================
            # ANALYTICAL RESPONSE ENGINE
            # =================================

            issues_found = []

            negative_count = 0

            positive_count = 0

            keywords_found = []
            products_found = []

            # ---------------------------------
            # ANALYZE RETRIEVED DOCS
            # ---------------------------------

            for doc in retrieved_docs:

                d = doc.lower()

                # -----------------------------
                # ISSUE TYPES
                # -----------------------------

                if "performance_issue" in d:
                    issues_found.append(
                        "charging reliability issues"
                    )

                if "durability_issue" in d:
                    issues_found.append(
                        "product durability problems"
                    )

                if "design_issue" in d:
                    issues_found.append(
                        "design-related complaints"
                    )

                if "usability_issue" in d:
                    issues_found.append(
                        "usability difficulties"
                    )

                # -----------------------------
                # SENTIMENT COUNTS
                # -----------------------------

                if "negative" in d:
                    negative_count += 1

                if "positive" in d:
                    positive_count += 1

                # -----------------------------
                # SIMPLE KEYWORD EXTRACTION
                # -----------------------------

                important_words = [

                    "broken",
                    "hot",
                    "charging",
                    "battery",
                    "slow",
                    "damage",
                    "cheap",
                    "crack",
                    "heat"
                ]
                product_match = re.search(
                    r'product id:\s*([a-z0-9]+)',
                    d,
                    re.IGNORECASE
                )

                if product_match:
                    products_found.append(
                        product_match.group(1).upper()
                    )

                for w in important_words:

                    if w in d:
                        keywords_found.append(w)



            # ---------------------------------
            # REMOVE DUPLICATES
            # ---------------------------------

            issues_found = list(
                set(issues_found)
            )

            keywords_found = list(
                set(keywords_found)
            )

            # =================================
            # BUILD RESPONSE
            # =================================

            answer = ""

            products_found = list(
                set(products_found)
            )

            if len(products_found) > 0:

                answer += "\nRelevant product IDs:\n\n"

                for p in products_found[:5]:
                    answer += f"• {p}\n"

            # ---------------------------------
            # ISSUE SUMMARY
            # ---------------------------------

            if len(issues_found) > 0:

                answer += (
                    "\nCommon issues identified "
                    "from retrieved reviews include:\n\n"
                )

                for i in issues_found:
                    answer += f"• {i}\n"

            # ---------------------------------
            # KEYWORDS
            # ---------------------------------

            if len(keywords_found) > 0:

                answer += "\nFrequently occurring complaint keywords:\n\n"

                for k in keywords_found[:5]:
                    answer += f"• {k}\n"

            # ---------------------------------
            # SENTIMENT SUMMARY
            # ---------------------------------

            answer += "\n"

            if negative_count > positive_count:

                answer += (
                    "Overall retrieved reviews "
                    "show predominantly negative "
                    "customer sentiment."
                )

            else:

                answer += (
                    "Overall retrieved reviews "
                    "show predominantly positive "
                    "customer sentiment."
                )

            # =================================
            # SHOW ANSWER FIRST
            # =================================

            st.markdown(
                "### Generated Answer"
            )

            st.success(answer)

            # =================================
            # SHOW RETRIEVED CONTEXT
            # =================================

            st.markdown(
                "### Retrieved Context"
            )

            for doc in retrieved_docs:
                st.info(doc)

else:

    st.info(
        "Upload files and click 'Process Files'"
    )