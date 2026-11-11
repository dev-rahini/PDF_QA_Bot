from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def read_pdf(pdf_path):

    text = ""

    reader = PdfReader(pdf_path)

    for page in reader.pages:
        pdf_text = page.extract_text()

        if pdf_text:
            text += pdf_text + "\n"

    return text


pdf_path = "documents/sample.pdf"

text = read_pdf(pdf_path)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors=embeddings.embed_query(chunks[0])
print(len(vectors))
print(vectors[:10])
vector_store = FAISS.from_texts(chunks,embedding=embeddings)
print("createdsuccessfully")
# -------------------------------------------------------------------
# 5. RETRIEVAL & GROUNDED PROMPT GENERATION
# -------------------------------------------------------------------
query = "What is the main topic or objective discussed in this document?"
print(f"\nUser Query: {query}")

# Perform similarity search to get top-3 closest semantic chunks
docs = vector_store.similarity_search(query, k=3)

print("\n--- Top Retrieved Context ---")
context = ""
for i, doc in enumerate(docs, 1):
    print(f"\n[Result {i}]:")
    print(doc.page_content.strip())
    context += doc.page_content + "\n\n"

# 6. Build the Augmented Prompt
rag_prompt = f"""You are a helpful AI assistant. Use strictly the provided context below to answer the user query.
If the answer is not present in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {query}
Answer:"""

print("\n--- Final Grounded RAG Prompt Sent to LLM ---")
print(rag_prompt)





