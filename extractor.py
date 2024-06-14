import os
from docxHelper import extractDocx
from pdfHelper import extractPdf
from instance import Instance
import nltk
import joblib
from preprocessing import preprocess_text
from modelAssembly import assembleModels
from bestModelAssembly import assembleBestModel
from graphs import plot_class_distribution
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def process_files_in_directory(root_directory):
    instances = []
    class_distribution = {}
    for folder_name, subfolders, filenames in os.walk(root_directory):
        class_name = os.path.basename(folder_name)
        print(class_name)
        if folder_name != root_directory:
            num_files = len(filenames)
            class_distribution[class_name] = num_files
        for filename in filenames:
            file_path = os.path.join(folder_name, filename)
            if filename.lower().endswith('.docx'):
                text = extractDocx(file_path)
            elif filename.lower().endswith('.pdf'):
                text = extractPdf(file_path)
            else:
                continue
            text = preprocess_text(text)
            instance = Instance(text, class_name)
            instances.append(instance)
    return instances, class_distribution

# Main function to process input text file and classify it
def main():
    directory = "data"
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    labeled_documents, class_distribution = process_files_in_directory(directory)

    plot_class_distribution(class_distribution)
    
    texts = [doc.text for doc in labeled_documents]
    labels = [doc.label for doc in labeled_documents]

    best_embedding, best_model = assembleModels(texts, labels)
    #best_embedding, best_model = 2, 0
    vectorizer, model = assembleBestModel(texts, labels, best_embedding, best_model)

    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    print("Model and Vectorizer saved and ready for analysis.")

    return

if __name__ == "__main__":
    main()
