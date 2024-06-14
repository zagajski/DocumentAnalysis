import fitz

def extractPdf(filepath):
    try:
        doc = fitz.open(filepath)
        extracted_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            extracted_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"An error occurred while extracting text from {filepath}: {e}")
        extracted_text = ""
    return extracted_text

def analyzePdf(filepath):
    extracted_text = extractPdf(filepath)
    
    # Word count
    word_count = len(extracted_text.split())
    
    # Character count (excluding spaces)
    char_count_no_spaces = len(extracted_text.replace(" ", ""))
    
    # Number of paragraphs
    paragraphs = extracted_text.split('\n')
    paragraph_count = len(paragraphs)
    
    # Page count
    doc = fitz.open(filepath)
    page_count = doc.page_count
    
    # Image extraction details
    images_info = []
    for page_index in range(page_count):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        images_info.append((page_index + 1, len(image_list)))
    
    # Table extraction details
    tables_info = []
    for page_index in range(page_count):
        page = doc[page_index]
        tabs = page.find_tables()
        tables_info.append((page_index + 1, len(tabs.tables)))
        for table_index, tab in enumerate(tabs.tables):
            df = tab.to_pandas()
            print(f"Table on page {page_index + 1}, table {table_index + 1}:")
            print(df)
    
    # Close the PDF document
    doc.close()
    
    # Print extracted information
    print(f"Page Count: {page_count}")
    print(f"Word Count: {word_count}")
    print(f"Character Count (excluding spaces): {char_count_no_spaces}")
    print(f"Paragraph Count: {paragraph_count}")
    
    # Print image details
    for page_num, image_count in images_info:
        if image_count != 0:
            print(f"Found {image_count} image(s) on page {page_num}")
    
    # Print table details
    for page_num, table_count in tables_info:
        if table_count != 0:
            print(f"Found {table_count} table(s) on page {page_num}")
    
    return extracted_text
