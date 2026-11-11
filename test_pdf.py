from pypdf import PdfReader

pdf_path="documents/sample.pdf"
read=PdfReader(pdf_path)
print("no of pages",len(read.pages))
print(read.pages[0].extract_text())
