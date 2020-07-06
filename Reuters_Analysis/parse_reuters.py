#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import collections
from keras.preprocessing.text import Tokenizer

def make_reuters_dataset(path=os.path.join('reuters21578'), min_samples_per_topic=15):

    wire_topics = []
    topic_counts = {}
    wire_bodies = []

    for fname in sorted(os.listdir(path)):
        if 'sgm' in fname:
            s = open(os.path.join(path, fname)).read()
            tag = '<TOPICS>'
            while tag in s:
                s = s[s.find(tag)+len(tag):]
                topics = s[:s.find('</')]
                if topics and '</D><D>' not in topics:
                    topic = topics.replace('<D>', '').replace('</D>', '')
                    wire_topics.append(topic)
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                else:
                    continue

                bodytag = '<BODY>'
                body = s[s.find(bodytag)+len(bodytag):]
                body = body[:body.find('</')]
                wire_bodies.append(body)

    # only keep most common topics
    items = list(topic_counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    kept_topics = set()
    for x in items:
        print(x[0] + ': ' + str(x[1])),
        if x[1] >= min_samples_per_topic:
            kept_topics.add(x[0])
    print
    print('-')
    print('Kept topics:', len(kept_topics))
    
    # filter wires with rare topics
    kept_wires = []
    labels = []
    topic_indexes = {}
    for t, b in zip(wire_topics, wire_bodies):
        if t in kept_topics:
            if t not in topic_indexes:
                topic_index = len(topic_indexes)
                topic_indexes[t] = topic_index
            else:
                topic_index = topic_indexes[t]

            labels.append(topic_index)
            kept_wires.append(b)

    print('Kept wires:', len(kept_wires))
    print('-')
    print('Topic mapping:', sorted(topic_indexes.items(), key=lambda x:x[1]))
    print('-')

    # vectorize wires
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(kept_wires)
    X = tokenizer.texts_to_sequences(kept_wires)

    print('Sanity check:')
    for w in ["banana", "oil", "chocolate", "the", "dsft"]:
        print('...index of', w, ':', tokenizer.word_index.get(w))
    print('text reconstruction:')
    reverse_word_index = dict([(v, k) for k, v in tokenizer.word_index.items()])
    print(' '.join(reverse_word_index[i] for i in X[10]))

    print collections.Counter(labels)
    dataset = (X, labels)
    print('-')

if __name__ == "__main__":
    make_reuters_dataset()

