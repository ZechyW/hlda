#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Tomotopy HLDA Test
import csv
import glob
import pathlib
import pickle
import re
import string
import time

import nltk
import tomotopy as tp
import tqdm

# from ipywidgets import widgets
# from IPython.core.display import HTML, display


# In[2]:


# Config

# Random seed for training runs
model_seed = 11399

# Dataset path
data_path = pathlib.Path("E:/Datasets/BBC/tech")
# data_path = pathlib.Path("E:/Datasets/stackoverflow_fb_kaggle/Test.csv")

# Pre-processing (need to retrain if changed, of course)
extra_tokenise = False
do_stemming = False

# # if ingest_and_save is False, will attempt to re-ingest raw data instead
# corpus_file = "corpus.bin"
# ingest_and_save = False

# If train_and_save is False, will attempt to load from model_file instead
model_file = "model.bin"
train_and_save = True

# Model type: lda | hlda
model_type = "hlda"
# How many clusters for lda
lda_k = 20
# How many levels for hlda
hlda_depth = 3

# Automated labelling
label_topics = True


# In[3]:

if model_type == "hlda":
    model = tp.HLDAModel(seed=model_seed, depth=hlda_depth)
if model_type == "lda":
    model = tp.LDAModel(seed=model_seed, k=lda_k)

# In[4]:

# Ingest data

# Stopwords
stopset = set(nltk.corpus.stopwords.words("english") + list(string.punctuation))
# Pronouns, titles
stopset.update(
    ["i", "i'm", "i'd", "i've", "i'll"]
    + ["she", "she's", "she'd", "she'll"]
    + ["he", "he's", "he'd", "he'll"]
    + ["they", "they're", "they'd", "they'll"]
    + ["mr", "dr"]
)
# Modals
stopset.update(["would", "could", "should", "shall", "can"])
# Corpus-specific
stopset.update(["will", "also", "said"])

# Stemming
if do_stemming:
    # Save a representative full token for each stem (for friendly display).
    # Each key -> 2nd level Dictionary of counts for full forms
    stem_to_word_count = {}
    stemmer = nltk.stem.snowball.SnowballStemmer("english")


# Pre-process doc
def doc_to_tokens(doc):
    # Remove unicode chars
    doc = doc.encode("ascii", "ignore").decode()

    # Case folding
    tokens = doc.casefold().split()

    # Preliminary removal of leading/trailing punctuation and stopwords
    tokens = [x.strip(string.punctuation) for x in tokens]
    tokens = [
        x for x in tokens if x and x not in stopset and not re.match(r"^(\W|\d)+$", x)
    ]

    # Extra tokenisation
    if extra_tokenise:
        clean = " ".join(tokens)
        tokens = nltk.word_tokenize(clean)
        # Secondary stopword cleaning
        tokens = [x.strip(string.punctuation) for x in tokens]
        tokens = [
            x
            for x in tokens
            if x and x not in stopset and not re.match(r"^(\W|\d)+$", x)
        ]

    # Stemming
    if do_stemming:
        new_tokens = []
        for token in tokens:
            stem = stemmer.stem(token)

            # Save friendly version
            if stem not in stem_to_word_count:
                stem_to_word_count[stem] = {}
            if token not in stem_to_word_count[stem]:
                stem_to_word_count[stem][token] = 1
            else:
                stem_to_word_count[stem][token] += 1

            new_tokens.append(stem)

        tokens = new_tokens

    # Add to model
    return tokens


# Ingest text files from directory (e.g., BBC dataset)
for file in glob.glob(f"{data_path}/*.txt"):
    with open(file) as f:
        doc = f.read()
        tokens = doc_to_tokens(doc)
        model.add_doc(tokens)

# Ingest from CSV (e.g., SO questions dataset)
# i = 0
# titles = []
# for i, row in tqdm.tqdm(
#     enumerate(csv.DictReader(open(data_path, "r", encoding="utf8")))
# ):
#     doc = row["Title"]
#     tokens = doc_to_tokens(doc)
#     model.add_doc(tokens)
#
# print(f"CSV rows ingested: {i}")

# Pickled version of SO question titles
# N.B.: The total *number* of documents is too crazy for Tomotopy -- Bigger
# documents shouldn't be a problem per se
# with open("so_titles.pkl", "rb") as f:
#     docs = pickle.load(f)
# for doc in tqdm.tqdm(docs[:1000000], total=1000000):
#     model.add_doc(doc_to_tokens(doc))

# Enron ham
# with open("enron_ham.pkl", "rb") as f:
#     ham = pickle.load(f)
#     docs = [f"{doc['subject']} {doc['content']}" for doc in ham]
#
# for doc in tqdm.tqdm(docs, total=len(docs)):
#     model.add_doc(doc_to_tokens(doc))

# Flatten stem_to_word_count to the full word with the highest count
if do_stemming:
    stem_to_word = {}

    for stem, counts in stem_to_word_count.items():
        highest = ["", 0]
        for word, count in counts.items():
            if count > highest[1]:
                highest = [word, count]

        stem_to_word[stem] = highest[0]

print("Data ingestion complete.")

# Word prior test
if model_type == "lda":
    model.set_word_prior("game", [10.0 if k == 0 else 0.1 for k in range(lda_k)])
    model.set_word_prior("console", [10.0 if k == 0 else 0.1 for k in range(lda_k)])
    model.set_word_prior("xbox", [10.0 if k == 0 else 0.1 for k in range(lda_k)])

# In[5]:

# Model training / Load model file
if train_and_save:
    model.train(0, workers=0)
    print(
        f"Num docs: {len(model.docs)}, Vocab size: {model.num_vocabs}, "
        f"Num words: {model.num_words}"
    )
    print(f"Removed top words: {model.removed_top_words}")

    train_batch = 50
    train_total = 500
    try:
        for i in range(0, train_total, train_batch):
            start_time = time.perf_counter()
            model.train(train_batch, workers=0)
            elapsed = time.perf_counter() - start_time
            print(
                f"Iteration: {i + train_batch}\tLog-likelihood: {model.ll_per_word}\tTime: "
                f"{elapsed:.3f}s",
                flush=True,
            )
    except KeyboardInterrupt:
        print("Stopping train sequence.")
    print(f"Saving to {model_file}.")
    model.save(model_file)
else:
    model = tp.HLDAModel.load(model_file)


# Results
# =======

# In[6]:


# Utils
def word_by_id(word_id):
    return model.vocabs[word_id]


# In[7]:

# Automated topic labelling
if label_topics:
    # extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    candidates = extractor.extract(model)
    # labeler = tp.label.FoRelevance(model, candidates, min_df=5, smoothing=1e-2, mu=0.25)
    labeler = tp.label.FoRelevance(model, candidates, min_df=3, smoothing=1e-2, mu=0.25)


# Results by topic
def print_topic(topic_id):
    # Labels
    if label_topics:
        labels = ", ".join(
            label for label, score in labeler.get_topic_labels(topic_id, top_n=5)
        )
        print(f"Suggested labels: {labels}")

    # Print this topic
    words_probs = model.get_topic_words(topic_id, top_n=10)
    words = [x[0] for x in words_probs]

    # Lookup stem -> most common form if necessary
    if do_stemming:
        words = [stem_to_word[x] for x in words]

    words = ", ".join(words)
    print(words)


def print_with_children(topic_id):
    # Print this topic
    words_probs = model.get_topic_words(topic_id, top_n=10)
    words = [x[0] for x in words_probs]

    # Lookup stem -> most common form if necessary
    if do_stemming:
        words = [stem_to_word[x] for x in words]

    # Prep for display
    words = ", ".join(words)
    level = model.level(topic_id)
    print(f"{'  ' * level}{level+ 1}: {words}")
    if label_topics:
        labels = ", ".join(
            label for label, score in labeler.get_topic_labels(topic_id, top_n=5)
        )
        print(f"{'  ' * level}   [Suggested: {labels}]")

    for child_id in model.children_topics(topic_id):
        if not model.is_live_topic(child_id):
            continue
        print_with_children(child_id)


def print_with_parents(topic_id):
    # Recursively print any parents first
    parent_id = model.parent_topic(topic_id)
    if parent_id >= 0:
        print_with_parents(parent_id)

    # Print this topic
    words_probs = model.get_topic_words(topic_id, top_n=10)
    words = [x[0] for x in words_probs]

    # Lookup stem -> most common form if necessary
    if do_stemming:
        words = [stem_to_word[x] for x in words]

    words = ", ".join(words)
    print(f"Level {model.level(topic_id)}: {words}")
    if label_topics:
        labels = ", ".join(
            label for label, score in labeler.get_topic_labels(topic_id, top_n=5)
        )
        print(f" -- Suggested Labels: {labels}")


for k in range(model.k):
    if model_type == "hlda":
        if not model.is_live_topic(k):
            continue
        if model.parent_topic(k) >= 0:
            # This is not a root node
            continue

        print_with_children(k)

    if model_type == "lda":
        print(f"[Topic {k}]")
        print_topic(k)
    print()


# # In[8]:
#
#
# # Interactive results by document
#
# colour_map = {0: "blue", 1: "red", 2: "green"}
#
# # Topic -> Level and Topic -> Top words mappings
# topic_to_level = {}
# topic_to_words = {}
# for k in range(model.k):
#     # Level
#     if not model.is_live_topic(k):
#         continue
#     topic_to_level[k] = model.level(k)
#
#     # Top words
#     word_probs = model.get_topic_words(k, top_n=10)
#     words = [x[0] for x in word_probs]
#
#     # Lookup stem -> most common form if necessary
#     if do_stemming:
#         words = [stem_to_word[x] for x in words]
#
#     topic_to_words[k] = words
#
# # Each document
# def show_doc(d=0):
#     doc = model.docs[d]
#     # Get unique doc topics -- Should be in ascending order of level after sorting
#     doc_topics = list(set(doc.topics))
#     doc_topics.sort()
#
#     # Header
#     for level in range(len(doc_topics)):
#         output = (
#             f"<h{level+1}><span style='color:{colour_map[level]}'>"
#             f"Topic {doc_topics[level]} (Level {level}): "
#             f"{', '.join(topic_to_words[doc_topics[level]])}"
#             f"</span></h{level+1}>"
#         )
#         display(HTML(output))
#
#     display(HTML("<hr/><h5>Processed Document</h5>"))
#
#     # Documents words
#     words = [word_by_id(x) for x in doc.words]
#     if do_stemming:
#         words = [stem_to_word[x] for x in words]
#
#     word_html = []
#     for word, topic in zip(words, doc.topics):
#         word_html.append(
#             f"<span style='color: {colour_map[topic_to_level[topic]]}'>{word}</span>"
#         )
#
#     display(HTML(" ".join(word_html)))
#
#
# # In[9]:
#
#
# widgets.interact(show_doc, d=(0, len(model.docs) - 1))
