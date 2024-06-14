import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(class_distribution):
    class_names = list(class_distribution.keys())
    class_counts = list(class_distribution.values())
    plt.figure(figsize=(14, 8))
    plt.bar(class_names, class_counts)
    plt.ylabel('number of instances')
    plt.title('Dataset Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_bars(values, y_label, title):
    models = ["SVM", "K-Nearest Neighbors", "Naive Bayes"]
    embeddings = ["GloVe", "Word2Vec", "TF-IDF", "Doc2Vec"]
    
    num_models = len(models)
    num_embeddings = len(embeddings)
    
    bar_width = 0.2
    x = np.arange(num_models) 

    fig, ax = plt.subplots()
    
    for i in range(num_embeddings):
        ax.bar(x + i * bar_width, [values_group[i] for values_group in values], bar_width, label=embeddings[i])
    
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (num_embeddings - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend(title='Embedding type')

    plt.show()

def plot_best_model_values(class_names, values, title):
    plt.figure(figsize=(14, 8))
    plt.bar(class_names, values)
    plt.ylabel('f1 score')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

