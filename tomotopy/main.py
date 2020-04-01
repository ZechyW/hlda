#!/usr/bin/env python
# coding: utf-8

# In[87]:


# Tomotopy HLDA Test
import tomotopy as tp

import glob
import string
import re
import nltk

from ipywidgets import widgets
from IPython.core.display import HTML, display


# In[54]:


# Config

# Random seed for training runs
model_seed = 7156

# Pre-processing (need to retrain if changed, of course)
extra_tokenise = True
do_stemming = True

# If train_and_save is False, will attempt to load from model_file instead
model_file = "bbc_model.bin"
train_and_save = False


# In[55]:


# Tomotopy model
# N.B.: Will be overwritten below if train_and_save is False
model = tp.HLDAModel(seed=model_seed, depth=3)


# In[56]:


# Ingest BBC data

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

# Iterate over data
for file in glob.glob("../bbc/tech/*.txt"):
    with open(file) as f:
        doc = f.read()

        # Remove unicode chars
        doc = doc.encode("ascii", "ignore").decode()

        # Case folding
        tokens = doc.casefold().split()

        # Preliminary removal of leading/trailing punctuation and stopwords
        tokens = [x.strip(string.punctuation) for x in tokens]
        tokens = [
            x
            for x in tokens
            if x and x not in stopset and not re.match(r"^(\W|\d)+$", x)
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
        model.add_doc(tokens)

# Flatten stem_to_word_count to the full word with the highest count
if do_stemming:
    stem_to_word = {}

    for stem, counts in stem_to_word_count.items():
        highest = ["", 0]
        for word, count in counts.items():
            if count > highest[1]:
                highest = [word, count]

        stem_to_word[stem] = highest[0]


# In[58]:


# Model training / Load model file
if train_and_save:
    for i in range(0, 500, 50):
        model.train(50)
        print("Iteration: {}\tLog-likelihood: {}".format(i, model.ll_per_word))
    print(f"Saving to {model_file}.")
    model.save(model_file)
else:
    model = tp.HLDAModel.load(model_file)


# Results
# =======

# In[67]:


# Utils
def word_by_id(word_id):
    return model.vocabs[word_id]


# In[59]:


# Results by topic
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


for k in range(model.k):
    if not model.is_live_topic(k):
        continue

    print(f"Topic {k}")
    print("=-=-=-=-=-=-=")
    print_with_parents(k)
    print()


# In[110]:


# Interactive results by document

colour_map = {0: "blue", 1: "red", 2: "green"}

# Topic -> Level and Topic -> Top words mappings
topic_to_level = {}
topic_to_words = {}
for k in range(model.k):
    # Level
    if not model.is_live_topic(k):
        continue
    topic_to_level[k] = model.level(k)

    # Top words
    word_probs = model.get_topic_words(k, top_n=10)
    words = [x[0] for x in word_probs]

    # Lookup stem -> most common form if necessary
    if do_stemming:
        words = [stem_to_word[x] for x in words]

    topic_to_words[k] = words

# Each document
def show_doc(d=0):
    doc = model.docs[d]
    # Get unique doc topics -- Should be in ascending order of level after sorting
    doc_topics = list(set(doc.topics))
    doc_topics.sort()

    # Header
    for level in range(len(doc_topics)):
        output = (
            f"<h{level+1}><span style='color:{colour_map[level]}'>"
            f"Topic {doc_topics[level]} (Level {level}): "
            f"{', '.join(topic_to_words[doc_topics[level]])}"
            f"</span></h{level+1}>"
        )
        display(HTML(output))

    display(HTML("<hr/><h5>Processed Document</h5>"))

    # Documents words
    words = [word_by_id(x) for x in doc.words]
    if do_stemming:
        words = [stem_to_word[x] for x in words]

    word_html = []
    for word, topic in zip(words, doc.topics):
        word_html.append(
            f"<span style='color: {colour_map[topic_to_level[topic]]}'>{word}</span>"
        )

    display(HTML(" ".join(word_html)))


# In[111]:


widgets.interact(show_doc, d=(0, len(model.docs) - 1))
