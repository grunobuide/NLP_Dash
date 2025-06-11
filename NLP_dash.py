import io
import streamlit as st
import pandas as pd
import spacy
import math
import emoji
import altair as alt
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

st.title("Gruno's Text Analysis Dashboard")

# Language selection
lang = st.selectbox("Select language of the text data", ["English", "Portuguese"])
model_name = "en_core_web_sm" if lang == "English" else "pt_core_news_sm"
nlp = spacy.load(model_name)

uploaded_file = st.file_uploader("Upload a CSV file with a column of texts", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_column = st.selectbox("Select the text column", df.columns)

    # --- Interactive Filtering ---
    filter_columns = [col for col in df.columns if col != text_column]
    if filter_columns:
        st.sidebar.header("Filter Data")
        for col in filter_columns:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 20:  # Only show filter if not too many unique values
                selected = st.sidebar.multiselect(f"Filter by {col}", unique_vals, default=list(unique_vals))
                df = df[df[col].isin(selected)]

    texts = df[text_column].astype(str).tolist()
    all_text = " ".join(texts)

    # Preprocessing options
    st.header("Text Preprocessing Options")
    do_lower = st.checkbox("Convert to lowercase", value=True)
    do_remove_punct = st.checkbox("Remove punctuation", value=True)
    do_lemmatize = st.checkbox("Lemmatize words", value=False)
    do_remove_stop = st.checkbox("Remove stopwords", value=True)
    do_include_emojis = st.checkbox("Consider emojis as words", value=True)



    # spaCy processing
    doc = nlp(all_text)

    # Stopword management
    st.subheader("Stopword Management")
    custom_stopwords = st.text_area("Add custom stopwords (comma separated)", "")
    # Build stopword set
    stopwords = set()
    if do_remove_stop:
        stopwords = nlp.Defaults.stop_words
        if custom_stopwords.strip():
            stopwords = stopwords.union({w.strip().lower() for w in custom_stopwords.split(",")})



    if st.button("Download current stopword list"):
        stopword_list = list(stopwords)
        stopword_str = "\n".join(sorted(stopword_list))
        st.download_button(
            label="Download Stopword List",
            data=stopword_str,
            file_name="stopwords.txt",
            mime="text/plain"
        )

    uploaded_stopwords = st.file_uploader("Upload a stopword list (one word per line)", type=["txt", "csv"])
    if uploaded_stopwords:
        uploaded_words = set(
            w.strip().lower()
            for w in uploaded_stopwords.read().decode("utf-8").splitlines()
            if w.strip()
        )
        stopwords = stopwords.union(uploaded_words)
        st.success(f"Loaded {len(uploaded_words)} stopwords from file.")

    # Tokenization and preprocessing (with emoji support)
    def is_emoji(token_text):
        return any(char in emoji.EMOJI_DATA for char in token_text)

    tokens = []
    for token in doc:
        token_text = token.text
        # Accept emoji as tokens if option is enabled, or apply normal filtering
        if do_include_emojis:
            if not (token.is_alpha or is_emoji(token_text)) and do_remove_punct:
                continue
        else:
            if not token.is_alpha and do_remove_punct:
                continue
        word = token.lemma_.lower() if do_lemmatize else token.text.lower() if do_lower else token.text
        # If it's an emoji and we are including emojis, keep as is (don't lowercase or lemmatize)
        if do_include_emojis and is_emoji(token_text):
            word = token_text
        if do_remove_stop and word in stopwords:
            continue
        tokens.append(word)

    filtered_tokens = tokens


    # Descriptive statistics
    st.header("Descriptive Statistics")
    st.write(f"Number of documents: {len(texts)}")
    st.write(f"Total characters: {len(all_text)}")
    st.write(f"Total words: {len([t for t in doc if t.is_alpha])}")

    # Lexical diversity
    lexical_diversity = len(set(filtered_tokens)) / len(filtered_tokens) if filtered_tokens else 0
    st.write(f"Lexical diversity (unique words / total words): {lexical_diversity:.3f}")

    # Longest word
    longest_word = max(filtered_tokens, key=len) if filtered_tokens else ""
    st.write(f"Longest word: {longest_word} ({len(longest_word)} characters)")

   # N-gram customization
    st.header("N-gram Analysis Options")
    ngram_n = st.slider("Select n-gram size (n)", min_value=2, max_value=5, value=2)
    ngram_top = st.slider("Number of top n-grams to display", min_value=5, max_value=30, value=10)

    # N-gram calculation
    ngrams_counter = Counter(zip(*[tokens[i:] for i in range(ngram_n)]))
    ngram_table = [
        {"N-gram": " ".join(gram), "Count": count}
        for gram, count in ngrams_counter.most_common(ngram_top)
    ]
    st.subheader(f"Top {ngram_top} {ngram_n}-grams")
    st.dataframe(ngram_table, use_container_width=True)

     # --- N-gram Interactive Visualization ---
    ngram_vis_df = pd.DataFrame(ngram_table)
    if not ngram_vis_df.empty:
        st.subheader(f"Top {ngram_top} {ngram_n}-grams")
        ngram_chart = alt.Chart(ngram_vis_df).mark_bar().encode(
            x=alt.X('N-gram', sort='-y'),
            y='Count',
            tooltip=['N-gram', 'Count']
        ).properties(width=600)
        st.altair_chart(ngram_chart, use_container_width=True)


    st.header("Keyword-in-Context (KWIC) Search")
    kwic_word = st.text_input("Enter a word or emoji for KWIC search:")
    window = st.slider("Context window size", min_value=1, max_value=5, value=2)
    if kwic_word:
        kwic_results = []
        for i, token in enumerate(tokens):
            if token == kwic_word:
                left = " ".join(tokens[max(0, i-window):i])
                right = " ".join(tokens[i+1:i+1+window])
                kwic_results.append({"Left Context": left, "Keyword": token, "Right Context": right})
        if kwic_results:
            kwic_df = pd.DataFrame(kwic_results)
            st.dataframe(kwic_df)
            # --- Export KWIC results as CSV ---
            st.download_button(
                label="Download KWIC Results as CSV",
                data=kwic_df.to_csv(index=False).encode('utf-8'),
                file_name=f'kwic_{kwic_word}.csv',
                mime='text/csv'
            )
        else:
            st.info(f"No occurrences of '{kwic_word}' found.")

    # WordCloud
    st.header("Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_tokens))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Vocabulary size
    vocab = set(filtered_tokens)
    st.write(f"Vocabulary size: {len(vocab)}")

    # Average word length
    avg_word_len = sum(len(word) for word in filtered_tokens) / len(filtered_tokens) if filtered_tokens else 0
    st.write(f"Average word length: {avg_word_len:.2f}")

    # Most common words (excluding stopwords)
    common_words = Counter(filtered_tokens).most_common(10)
    common_words_table = [{"Word": word, "Count": count} for word, count in common_words]
    st.subheader("Top 10 Most Common Words (excluding stopwords)")
    st.dataframe(common_words_table, use_container_width=True)

    # Most common words (excluding stopwords)
    common_words = Counter(filtered_tokens).most_common(20)
    common_words_df = pd.DataFrame(common_words, columns=["Word", "Count"])
    st.subheader("Top 20 Most Common Words (Interactive)")
    chart = alt.Chart(common_words_df).mark_bar().encode(
        x=alt.X('Word', sort='-y'),
        y='Count',
        tooltip=['Word', 'Count']
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)

    # Hapax legomena (words that occur only once)
    hapaxes = [word for word, count in Counter(filtered_tokens).items() if count == 1]
    st.write(f"Number of hapax legomena (unique words): {len(hapaxes)}")

    # Sentence statistics
    sentences = list(doc.sents)
    st.write(f"Number of sentences: {len(sentences)}")
    avg_sentence_len = sum(len([t for t in sent if t.is_alpha]) for sent in sentences) / len(sentences) if sentences else 0
    st.write(f"Average sentence length (in words): {avg_sentence_len:.2f}")

    # POS Tagging statistics
    st.header("Part-of-Speech (POS) Tagging")
    pos_counts = Counter([token.pos_ for token in doc if token.is_alpha])
    pos_table = [{"POS": pos, "Count": count} for pos, count in pos_counts.most_common(10)]
    st.dataframe(pos_table, use_container_width=True)

    # Named Entity Recognition (NER)
    st.header("Named Entity Recognition (NER)")
    if doc.ents:
        ent_labels = [ent.label_ for ent in doc.ents]
        ent_types = Counter(ent_labels)
        ent_table = [{"Entity Type": label, "Count": count} for label, count in ent_types.most_common()]
        st.dataframe(ent_table, use_container_width=True)

        # Show top 10 named entities by frequency
        ent_texts = [ent.text for ent in doc.ents]
        ent_texts_counter = Counter(ent_texts)
        top_entities = [{"Entity": ent, "Count": count} for ent, count in ent_texts_counter.most_common(10)]
        st.subheader("Top 10 Named Entities")
        st.dataframe(top_entities, use_container_width=True)
    else:
        st.info("No named entities found in the current selection.")
    
    if doc.ents:
        ent_texts_counter = Counter([ent.text for ent in doc.ents])
        ent_texts_df = pd.DataFrame(ent_texts_counter.most_common(20), columns=["Entity", "Count"])
        if not ent_texts_df.empty:
            st.subheader("Top 20 Named Entities")
            ent_chart = alt.Chart(ent_texts_df).mark_bar().encode(
                x=alt.X('Entity', sort='-y'),
                y='Count',
                tooltip=['Entity', 'Count']
            ).properties(width=600)
            st.altair_chart(ent_chart, use_container_width=True)

    # # --- Time Series Analysis ---
    # date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower()]
    # if date_columns:
    #     st.header("Time Series Analysis")
    #     date_col = st.selectbox("Select a date/time column for time series analysis:", date_columns)
    #     # Try to convert to datetime if not already
    #     df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    #     df_time = df.dropna(subset=[date_col])
    #     # Group by date (day)
    #     docs_per_day = df_time.groupby(df_time[date_col].dt.date).size()
    #     st.subheader("Documents per Day")
    #     st.line_chart(docs_per_day)
    #     # Optional: Frequency of a word over time
    #     word_freq_word = st.text_input("Track frequency of a word over time (leave blank to skip):", "")
    #     if word_freq_word:
    #         freq_per_day = df_time[text_column].apply(lambda x: word_freq_word in str(x).split()).groupby(df_time[date_col].dt.date).sum()
    #         st.line_chart(freq_per_day.rename(f"Occurrences of '{word_freq_word}'"))

        # --- Topic Modeling ---
    st.header("Topic Modeling (LDA)")
    num_topics = st.slider("Number of topics", min_value=2, max_value=10, value=3)
    num_words = st.slider("Number of top words per topic", min_value=3, max_value=15, value=5)

    # Use filtered tokens (without stopwords) to reconstruct documents for LDA
    docs_for_lda = []
    for doc_text in texts:
        doc_spacy = nlp(doc_text)
        tokens_lda = [
            w.text.lower()
            for w in doc_spacy
            if w.is_alpha and (not do_remove_stop or w.text.lower() not in stopwords)
        ]
        docs_for_lda.append(" ".join(tokens_lda))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs_for_lda)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    topic_words = []
    for idx, topic in enumerate(lda.components_):
        top_features = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topic_words.append({"Topic": f"Topic {idx+1}", "Top Words": ", ".join(top_features)})

    topic_words_df = pd.DataFrame(topic_words)
    st.dataframe(topic_words_df)

        # --- Topic Modeling Interactive Visualization ---
    if not topic_words_df.empty:
        st.subheader("LDA Topics")
        # Explode the top words for each topic for visualization
        topic_exploded = topic_words_df.copy()
        topic_exploded["Top Words"] = topic_exploded["Top Words"].str.split(", ")
        topic_exploded = topic_exploded.explode("Top Words")
        topic_chart = alt.Chart(topic_exploded).mark_bar().encode(
            x=alt.X('Top Words', sort='-y'),
            y=alt.Y('Topic', sort=None),
            color='Topic',
            tooltip=['Topic', 'Top Words']
        ).properties(width=600)
        st.altair_chart(topic_chart, use_container_width=True)


    # --- Download Section reflecting customizable n-gram range ---
    st.subheader("Download Results")
    # Create DataFrame with ALL n-grams for the selected n
    all_ngram_table = [
        {"N-gram": " ".join(gram), "Count": count}
        for gram, count in ngrams_counter.items()
    ]
    ngram_df = pd.DataFrame(all_ngram_table)
    pos_df = pd.DataFrame(pos_table)
    st.download_button(
        label=f"Download ALL {ngram_n}-grams as CSV",
        data=ngram_df.to_csv(index=False).encode('utf-8'),
        file_name=f'all_{ngram_n}grams.csv',
        mime='text/csv'
    )
    # Create DataFrame for POS table
    st.download_button(
        label="Download POS Table as CSV",
        data=pos_df.to_csv(index=False).encode('utf-8'),
        file_name='pos_table.csv',
        mime='text/csv'
    )

    # Downloadable wordcloud
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download WordCloud as PNG",
        data=buf.getvalue(),
        file_name="wordcloud.png",
        mime="image/png"
    )

    # Download LDA topics as CSV
    st.download_button(
        label="Download LDA Topics as CSV",
        data=topic_words_df.to_csv(index=False).encode('utf-8'),
        file_name='lda_topics.csv',
        mime='text/csv'
    )
