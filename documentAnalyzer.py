import argparse
import os
import numpy as np
import joblib
from docxHelper import analyzeDocx
from pdfHelper import analyzePdf
from preprocessing import preprocess_text
from collections import Counter

def find_common_words(preprocessed_text1, preprocessed_text2):
    # Tokenize the preprocessed texts
    tokens1 = preprocessed_text1.split()
    tokens2 = preprocessed_text2.split()
    
    # Count the occurrences of each word in both texts
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Find the common words in both texts
    common_words = set(counter1.keys()) & set(counter2.keys())
    
    # Sum the counts of these common words from both texts
    common_word_counts = {word: counter1[word] + counter2[word] for word in common_words}
    
    # Sort the common words by their total counts in descending order
    sorted_common_words = sorted(common_word_counts.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_common_words

def print_top_common_words(common_words, filename1, filename2, top_n=10):
    print(f"\nTop {top_n} common words in file 1 ({filename1}) and file 2 ({filename2}):")
    print(f"{'Word':<20} {'Count':<10}")
    print("-" * 30)
    for word, count in common_words[:top_n]:
        print(f"{word:<20} {count:<10}")


# Function to read text from a file
def read_text_from_file(filepath):
    try:
        if filepath.lower().endswith('.docx'):
            text = analyzeDocx(filepath)
        elif filepath.lower().endswith('.pdf'):
            text = analyzePdf(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a .docx or .pdf file.")
    except Exception as e:
        print(f"An error occurred while reading text from {filepath}: {e}")
        text = ""
    return text

# Main function to process input text file and classify it
def main():
    parser = argparse.ArgumentParser(description='Classify input text file using pre-trained model.')
    parser.add_argument('filenames', type=str, nargs='*', help='Name(s) of the input text file(s)')
    args = parser.parse_args()

    try:
        vectorizer = joblib.load("vectorizer.joblib")
        loaded_model = joblib.load("model.joblib")
    except Exception as e:
        print(f"An error occurred while loading the model or vectorizer: {e}")
        return

    if len(args.filenames) == 1:
        filename = args.filenames[0]
        filepath = os.path.join(os.path.dirname(__file__), filename)

        try:
            input_text = read_text_from_file(filepath)
            if not input_text:
                raise ValueError(f"No text extracted from file {filename}")

            preprocessed_text = preprocess_text(input_text)
            vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
            prediction = loaded_model.predict(vectorized_text)
            print(f"\nClassification of file {filename}:", prediction)
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    elif len(args.filenames) == 2:
        filename1, filename2 = args.filenames
        filepath1 = os.path.join(os.path.dirname(__file__), filename1)
        filepath2 = os.path.join(os.path.dirname(__file__), filename2)

        try:
            input_text1 = read_text_from_file(filepath1)
            input_text2 = read_text_from_file(filepath2)
            if not input_text1 or not input_text2:
                raise ValueError("No text extracted from one or both files")

            preprocessed_text1 = preprocess_text(input_text1)
            preprocessed_text2 = preprocess_text(input_text2)

            common_words = find_common_words(preprocessed_text1, preprocessed_text2)
            print_top_common_words(common_words, filename1, filename2)

            vectorized_text1 = vectorizer.transform([preprocessed_text1]).toarray()
            vectorized_text2 = vectorizer.transform([preprocessed_text2]).toarray()

            prediction1 = loaded_model.predict(vectorized_text1)
            prediction2 = loaded_model.predict(vectorized_text2)

            print(f"\nClassification of file 1 ({filename1}):", prediction1)
            print(f"Classification of file 2 ({filename2}):", prediction2)
        except Exception as e:
            print(f"An error occurred while processing files {filename1} and {filename2}: {e}")

    else:
        print("Unsupported number of files. Please provide one or two filenames.")
if __name__ == "__main__":
    main()