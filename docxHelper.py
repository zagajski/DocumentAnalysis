import docx2txt

def extractDocx(filepath):
    try:
        extracted_text = docx2txt.process(filepath)
    except Exception as e:
        print(f"An error occurred while extracting text from {filepath}: {e}")
        extracted_text = ""
    return extracted_text

def analyzeDocx(filepath):
    extracted_text = extractDocx(filepath)
    
    # Word count
    word_count = len(extracted_text.split())
    
    # Character count (excluding spaces)
    char_count_no_spaces = len(extracted_text.replace(" ", ""))
    
    # Number of paragraphs
    paragraphs = extracted_text.split('\n')
    paragraph_count = len(paragraphs)
    
    # Print extracted information
    print(f"Word Count: {word_count}")
    print(f"Character Count (excluding spaces): {char_count_no_spaces}")
    print(f"Paragraph Count: {paragraph_count}")
    
    return extracted_text