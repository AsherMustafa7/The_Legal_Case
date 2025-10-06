# scripts/cluster_lda.py
import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def load_cases(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def main(args):
    df = load_cases(args.input)
    texts = df["summary"].fillna("").tolist()

    # ✅ Remove stopwords (important for meaningful topics)
    vectorizer = CountVectorizer(max_features=2000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    results = []
    for n_topics in range(2, 15):  # test topic counts 2–14
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)

        perplexity = lda.perplexity(X)
        log_likelihood = lda.score(X)

        # Collect top words per topic
        topic_keywords = []
        for i, topic in enumerate(lda.components_):
            top_terms = [feature_names[idx] for idx in topic.argsort()[-10:]]
            topic_keywords.append(", ".join(top_terms))

        results.append((n_topics, perplexity, log_likelihood, topic_keywords))

        print(f"\nTopics={n_topics} | Perplexity={perplexity:.2f} | Log-Likelihood={log_likelihood:.2f}")
        for i, words in enumerate(topic_keywords):
            print(f"   Topic {i}: {words}")

    # Save results to Excel
    rows = []
    for n_topics, perplexity, log_likelihood, keywords in results:
        for i, topic in enumerate(keywords):
            rows.append((n_topics, perplexity, log_likelihood, i, topic))

    pd.DataFrame(rows, columns=["n_topics", "perplexity", "log_likelihood", "topic_id", "keywords"])\
        .to_excel(args.output, index=False)

    print(f"\n✅ Evaluation results saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
