from embeddings import embeddingWith
from models import modelSelector
from graphs import plot_best_model_values

def assembleBestModel(texts:list, labels:list, i, j):
    
    X, embedding_type, vectorizer = embeddingWith(i, texts, True)
    model_type, accuracyscore, classificationreport, model = modelSelector(X, labels, j, embedding_type, True)
    
    class_names = list(classificationreport.keys())[:-3]
    precision_by_class = [classificationreport[class_name]['precision'] for class_name in class_names]
    recall_by_class = [classificationreport[class_name]['recall'] for class_name in class_names]
    f1_scores_by_class = [classificationreport[class_name]['f1-score'] for class_name in class_names]

    plot_best_model_values(class_names, precision_by_class, f'Class-wise Precisions of {model_type} with {embedding_type}')
    plot_best_model_values(class_names, recall_by_class, f'Class-wise Recall of {model_type} with {embedding_type}')
    plot_best_model_values(class_names, f1_scores_by_class, f'Class-wise F1 Scores of {model_type} with {embedding_type}')

    return vectorizer, model