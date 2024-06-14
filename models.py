import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler  

def train_svm(X_train, y_train):
    print("started svm")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model, 'SVM'

def train_knn(X_train, y_train, n_neighbors=5):
    print("started knn")
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model, 'K-Nearest Neighbors'

def train_naive_bayes(X_train, y_train):
    print("started nb")
    model = MultinomialNB()
    #if np.any(X_train < 0):
    #    X_train = X_train + abs(np.min(X_train))
    #model.fit(X_train, y_train)
    scaler = MinMaxScaler()
    X_train_transformed = scaler.fit_transform(X_train)
    model.fit(X_train_transformed, y_train)
    return model, 'Naive Bayes'

def evaluate_model(model, X_test, y_test, model_type, embedding_type):
    y_pred = model.predict(X_test)
    accuracyscore = accuracy_score(y_test, y_pred)
    print(f'\n[{embedding_type}, {model_type}] Accuracy: {accuracyscore}')
    classificationreport = classification_report(y_test, y_pred, output_dict=False, zero_division=0)
    print(f'[{embedding_type}, {model_type}] Classification report:\n{classificationreport}')
    macrof1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f'[{embedding_type}, {model_type}] Macro F1-score: {macrof1}')
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return accuracyscore, report_dict


def modelSelector(X, y, index, embedding_type, return_trained_model:bool):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    if index == 0:
        model, model_type = train_svm(X_train, y_train)
    elif index == 1:
        model, model_type = train_knn(X_train, y_train)
    elif index == 2:
        model, model_type = train_naive_bayes(X_train, y_train)
    else:
        raise ValueError("Invalid index. Choose a number between 0 and 2.")

    accuracyscore, report_dict = evaluate_model(model, X_test, y_test, model_type, embedding_type)
    if not return_trained_model:
        return model_type, accuracyscore, report_dict
    else:
        return model_type, accuracyscore, report_dict, model
