from pypdf import PdfReader
with open("paper_text.txt", "w", encoding="utf-8") as out:
    try:
        reader = PdfReader("paper.pdf")
        for i, page in enumerate(reader.pages):
            out.write(f"--- PAGE {i+1} ---\n")
            out.write(page.extract_text() + "\n")
    except Exception as e:
        out.write(f"Error reading PDF: {e}")
