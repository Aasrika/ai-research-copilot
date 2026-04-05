import os
import json
import fitz                          # PyMuPDF — faster and cleaner than pypdf2
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()  # pulls OPENAI_API_KEY from your .env file



def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.
    
    WHY PAGE-BY-PAGE?
    We preserve page numbers as metadata. Later, when the system says
    "this answer comes from page 4", that's coming from here.
    
    WHY PyMuPDF (fitz)?
    - Handles multi-column layouts better than pypdf2
    - Preserves more formatting cues
    - Much faster on large papers
    
    Returns: list of dicts, one per page:
        [{"page": 1, "text": "...", "source": "paper.pdf"}, ...]
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # "text" mode: plain text extraction
        
        # Skip nearly-empty pages (figures, blank pages, etc.)
        if len(text.strip()) < 50:
            continue
        
        pages.append({
            "page": page_num,
            "text": text.strip(),
            "source": Path(pdf_path).name
        })
    
    doc.close()
    print(f"✅ Parsed {len(pages)} pages from {Path(pdf_path).name}")
    return pages



def chunk_pages(pages: list[dict], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Document]:
    """
    Split pages into overlapping chunks and wrap in LangChain Document objects.
    
    WHY DO WE CHUNK AT ALL?
    Embeddings work best on short, focused passages. A whole page (say, 600 words)
    might talk about 5 different things — the embedding averages all of them,
    making it match nothing well. A focused 150-word chunk on "attention mechanism"
    will match the query "how does self-attention work?" almost perfectly.
    
    WHY OVERLAP?
    If a key sentence sits at the boundary between two chunks, overlap ensures
    it appears fully in at least one chunk. Without overlap, you'd lose context
    right at the split points.
    
    chunk_size=800 tokens (≈600 words): good default for research papers.
    chunk_overlap=150: ~18% overlap is a reasonable starting point.
    
    WHY RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks → sentence breaks → word breaks,
    in that priority order. This keeps semantically coherent units together
    much better than splitting every N characters blindly.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
        length_function=len,
    )
    
    all_chunks = []
    
    for page in pages:
        # Split the page text
        splits = splitter.split_text(page["text"])
        
        for i, chunk_text in enumerate(splits):
            # LangChain's Document object = text + metadata dict
            # Metadata is CRUCIAL — it gets stored alongside embeddings in FAISS
            # and returned with every search result
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": i,
                    # We'll add section detection in Phase 2
                }
            )
            all_chunks.append(doc)
    
    print(f"✅ Created {len(all_chunks)} chunks "
          f"(avg {sum(len(d.page_content) for d in all_chunks) // len(all_chunks)} chars each)")
    return all_chunks



def get_embedding_model(use_openai: bool = True):
    """
    WHAT IS AN EMBEDDING?
    An embedding model converts text → a high-dimensional vector (e.g., 1536 numbers).
    Texts with similar *meaning* get vectors that point in similar directions.
    
    Cosine similarity between two vectors ≈ semantic similarity between two texts.
    This is why "cardiac arrest" and "heart attack" have similar embeddings even
    though they share no words.
    
    TWO OPTIONS:
    
    1. OpenAI text-embedding-3-small (use_openai=True)
       - 1536 dimensions
       - Very high quality, especially for scientific text
       - Costs ~$0.02 per million tokens → practically free for a research project
       - Requires OPENAI_API_KEY in .env
    
    2. Sentence Transformers (use_openai=False) — FREE, runs locally
       - all-MiniLM-L6-v2: 384 dimensions, fast, decent quality
       - Good fallback if you don't want to use OpenAI credits
    
    For this project, OpenAI embeddings are recommended because scientific
    vocabulary is better represented in their training data.
    """
    if use_openai:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



def build_vector_index(chunks: list[Document], embedding_model, save_dir: str = "data/indices") -> FAISS:
    """
    Build a FAISS vector index from chunks and save it to disk.
    
    WHAT IS FAISS?
    Facebook AI Similarity Search. It stores all your chunk embeddings and
    lets you find the K most similar ones to any query embedding in milliseconds,
    even with millions of vectors.
    
    HOW IT WORKS (simplified):
    1. Each chunk → embedding vector (1536 numbers)
    2. FAISS organizes them in an index structure optimized for similarity search
    3. At query time: embed the query → find nearest vectors → return those chunks
    
    WHY SAVE IT?
    Embedding 200 chunks costs ~$0.001. But you don't want to re-embed every time
    you restart the app. Save the index, load it on startup.
    """
    print("⏳ Building FAISS index (this calls the embedding API)...")
    
    # This single line: embeds all chunks + builds the FAISS index
    vector_store = FAISS.from_documents(chunks, embedding_model)
    
    # Save to disk
    os.makedirs(save_dir, exist_ok=True)
    vector_store.save_local(save_dir)
    print(f"✅ Index saved to {save_dir}/")
    
    return vector_store


def load_vector_index(embedding_model, index_dir: str = "data/indices") -> FAISS:
    """Load a previously saved FAISS index from disk."""
    return FAISS.load_local(
        index_dir, 
        embedding_model,
        allow_dangerous_deserialization=True  # required flag in newer LangChain
    )



def retrieve_chunks(query: str, vector_store: FAISS, k: int = 5) -> list[Document]:
    """
    Given a user query, find the k most relevant chunks.
    
    WHY k=5?
    You want enough context to answer the question, but not so much that
    you fill the LLM's context window with noise. 5 chunks × ~800 chars
    = ~600 tokens of context, very manageable.
    
    This returns Documents with similarity scores. We'll use scores later
    in the Critic Agent to flag low-confidence retrievals.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # results = list of (Document, score) tuples
    # In FAISS with L2 distance: LOWER score = MORE similar (it's a distance)
    # With cosine similarity: HIGHER score = MORE similar
    # LangChain normalizes this for you, but good to know.
    
    print(f"\n🔍 Top {k} chunks for: '{query[:60]}...'")
    for doc, score in results:
        print(f"   [{doc.metadata['source']} p.{doc.metadata['page']}] score={score:.3f}")
    
    return [doc for doc, _ in results]



def process_paper(pdf_path: str, use_openai: bool = True) -> FAISS:
    """End-to-end: PDF → searchable index."""
    pages = parse_pdf(pdf_path)
    chunks = chunk_pages(pages)
    embeddings = get_embedding_model(use_openai=use_openai)
    vector_store = build_vector_index(chunks, embeddings)
    return vector_store


if __name__ == "__main__":
    # QUICK TEST — drop any PDF into data/papers/ and run this
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/papers/test_paper.pdf"
    
    store = process_paper(pdf, use_openai=False)
    
    # Test retrieval
    query = "What is the main contribution of this paper?"
    chunks = retrieve_chunks(query, store, k=3)
    
    print("\n📄 Retrieved chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} (p.{chunk.metadata['page']}) ---")
        print(chunk.page_content[:300] + "...")