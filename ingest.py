import re
import json
import fitz

PDF_PATH = "constitution_nepal_2015.pdf"

OUTPUT_PATH = "articles.json"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        full_text += text + "\n" 

        if (page_num % 20 == 0):
            print(f"...processed {page_num} pages")
    doc.close()
    print(f"Extracted: {len(full_text)}")

    return full_text

def split_into_articles(full_text):
    pattern = re.compile(r'(?m)^\s*(\d+)\.\s*$')

    matches = list(pattern.finditer(full_text))

    articles = []

    for i in range(len(matches)):
        articles_number = int(matches[i].group(1))
        start = matches[i].start()

        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(full_text)

        article_text = full_text[start:end].strip()

        if len(article_text) > 50:
            articles.append(
                {
                    "article_number": articles_number,
                    "article_text": article_text,
                    "source": "Constitution of Nepal 2015",
                    "language": "en",
                    "chunk_id": f"constitution_article_{articles_number}"
                }
            )
    print(f"Found {len(articles)} articles.\n")

    return articles

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\(?:\d+|(\d+\))\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    return text

def main():
    # 1. Extract all text from the PDF
    full_text = extract_text_from_pdf(PDF_PATH)

    # 2. Split into articles
    articles = split_into_articles(full_text)

    # 3. Clean each article's text
    for article in articles:
        article['article_text'] = clean_text(article['article_text'])

    # 4. Show a preview of what we found
    for article in articles[:3]:
        preview = article["article_text"][:150].replace("\n", " ")
        print(f"Article {article['article_number']}: {preview}...")
        print()

    # 5. Save to JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii = False, indent = 2)

if __name__ == "__main__":
    main()