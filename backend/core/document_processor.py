import os, re, fitz
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from backend.core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    SECTION_KEYWORDS, PAPERS_DIR, INDICES_DIR
)

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_section(text: str) -> str:
    text_lower = text.lower()[:300]

    heading_match = re.match(r"^(\d+\.?\s+)?([A-Z][A-Za-z\s]{2,30})\n", text)
    if heading_match:
        heading = heading_match.group(2).lower().strip()
        for section, keywords in SECTION_KEYWORDS.items():
            if any(kw in heading for kw in keywords):
                return section

    for section, keywords in SECTION_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return section

    return "body"


# ─────────────────────────────────────────────────────────────────────────────
# PDF PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    name = Path(pdf_path).stem

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) < 50:
            continue

        pages.append({
            "page": page_num,
            "text": text,
            "source": Path(pdf_path).name,
            "paper_title": name,
            "total_pages": len(doc),
        })

    doc.close()
    print(f"  📄 {Path(pdf_path).name}: {len(pages)} pages extracted")
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page in pages:
        splits = splitter.split_text(page["text"])

        for i, text in enumerate(splits):
            section = detect_section(text)

            chunks.append(Document(
                page_content=text,
                metadata={
                    "source": page["source"],
                    "paper_title": page["paper_title"],
                    "page": page["page"],
                    "total_pages": page["total_pages"],
                    "chunk_index": i,
                    "section": section,
                    "char_count": len(text),
                }
            ))

    print(f"  🧩 {len(chunks)} chunks created")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS (LOCAL — FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────────────────────
# INDEX MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_or_update_index(pdf_paths: list[str], index_dir: str = str(INDICES_DIR)) -> FAISS:
    embeddings = get_embedding_model()

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    index_file = index_path / "index.faiss"

    all_chunks = []

    for pdf in pdf_paths:
        print(f"\n⏳ Processing: {Path(pdf).name}")
        pages = parse_pdf(pdf)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    print(f"\n⏳ Embedding {len(all_chunks)} chunks...")
    new_store = FAISS.from_documents(all_chunks, embeddings)

    if index_file.exists():
        print("📂 Existing index found — merging...")
        existing = FAISS.load_local(str(index_path), embeddings,
                                   allow_dangerous_deserialization=True)
        existing.merge_from(new_store)
        existing.save_local(str(index_path))
        print("✅ Index updated (merged)")
        return existing
    else:
        new_store.save_local(str(index_path))
        print(f"✅ New index saved to {index_path}/")
        return new_store


def load_index() -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(str(INDICES_DIR), embeddings,
                           allow_dangerous_deserialization=True)


def get_indexed_papers(index_dir: str = str(INDICES_DIR)) -> list[str]:
    try:
        store = load_index()
        titles = set()

        for doc_id in store.docstore._dict:
            doc = store.docstore._dict[doc_id]
            titles.add(doc.metadata.get("paper_title", "Unknown"))

        return sorted(titles)

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdfs = sys.argv[1:] if len(sys.argv) > 1 else []

    if not pdfs:
        print("Usage: python document_processor.py paper1.pdf paper2.pdf ...")
    else:
        build_or_update_index(pdfs)
        print("\n📚 Papers in index:", get_indexed_papers())