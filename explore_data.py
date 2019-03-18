import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_num_classes(labels):
    num_classes = max(labels) + 1
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s)')

    if num_classes <= 1:
        raise ValueError('Invalid number of labels')

    return num_classes

def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Sample Length distribution')
    plt.show()

"""
train_data, test_data = load_imdb_sentiment_analysis_dataset()
train_text, train_labels = train_data[0], train_data[1]
#print(train_text[:2])
#print(train_labels[:2])

print(float(len(train_text)) / float(get_num_words_per_sample(train_text)))
#print("ratio = ", get_ratio_num_samples_words_per_sample(train_text))

#plot_sample_length_distribution(train_text)
"""