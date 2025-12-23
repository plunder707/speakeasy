#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wiki_governor.py - Resonance Governor with LanceDB Vector Retrieval
Features:
- Full Wikipedia articles (no limits)
- Full arXiv abstracts
- Actual web page scraping (not just snippets)
- LanceDB vector database for semantic retrieval
- Pulls 50k-100k+ chars of context when available
- Vision capabilities: Image understanding with LLaVA 13B
- Automatic model selection (Llama3 for text, LLaVA for visual queries)
- Diagram code generation for complex scientific topics
"""

import sys
import wikipedia
import re
import urllib.parse
import feedparser
import logging
import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ddgs import DDGS
import trafilatura
import requests
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from pypdf import PdfReader

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("VectorGovernor")

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("trafilatura").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

try:
    import lancedb
except ImportError:
    print("❌ Critical: 'lancedb' not found. Run `pip install lancedb`.")
    sys.exit(1)

try:
    import ollama
except ImportError:
    print("❌ Critical: 'ollama' not found. Run `pip install ollama`.")
    sys.exit(1)

try:
    from resonance_governor import OllamaResonanceWrapper
except ImportError:
    print("❌ Critical: 'resonance_governor.py' not found.")
    sys.exit(1)

# --- CONFIGURATION ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
TOP_K_CHUNKS = 30  # Number of chunks to retrieve (increased for diversity)
MAX_WEB_SCRAPES = 30  # Max web pages to scrape
MAX_ARXIV_PAPERS = 20  # Max arXiv papers to fetch

# Visualization detection keywords
VISUAL_KEYWORDS = [
    "cycle", "pathway", "reaction", "structure", "circuit",
    "diagram", "network", "molecule", "mechanism", "photosystem",
    "wavefunction", "orbital", "topology", "architecture", "flowchart",
    "schematic", "graph", "tree", "map", "layout", "pipeline"
]

# Image storage configuration
IMAGE_DIR = Path("./scraped_images")
IMAGE_DIR.mkdir(exist_ok=True)
IMAGE_CACHE_FILE = IMAGE_DIR / "cache.json"
MIN_IMAGE_SIZE = 200  # Skip images smaller than 200x200 (logos, icons)

# URL patterns to skip (non-content images)
SKIP_IMAGE_PATTERNS = [
    'logo', 'icon', 'favicon', 'banner', 'button', 'badge',
    'avatar', 'thumb', 'social', 'share', 'ad', 'sponsor',
    'header', 'footer', 'nav', 'menu', 'widget', '1x1'
]

@dataclass
class Document:
    """Represents a scraped document"""
    title: str
    content: str
    source: str
    url: str
    doc_type: str  # 'wikipedia', 'arxiv', 'web'

# --- IMAGE CACHE MANAGEMENT ---

def load_image_cache() -> dict:
    """Load global image cache from disk."""
    if IMAGE_CACHE_FILE.exists():
        try:
            with open(IMAGE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load image cache: {e}")
    return {}

def save_image_cache(cache: dict) -> None:
    """Save global image cache to disk."""
    try:
        with open(IMAGE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save image cache: {e}")

def get_image_hash(image_data: bytes) -> str:
    """Generate MD5 hash of image content for deduplication."""
    return hashlib.md5(image_data).hexdigest()

def save_session_metadata(session_id: str, query: str, image_metadata: list) -> None:
    """Save session metadata to JSON file."""
    import datetime

    session_dir = IMAGE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "query": query,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_images": len(image_metadata),
        "images": image_metadata
    }

    metadata_file = session_dir / "metadata.json"
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Session metadata saved to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save session metadata: {e}")

class VectorKnowledgeBase:
    """Manages document chunking, embedding, and retrieval with LanceDB"""

    def __init__(self, db_path: str = "./lancedb"):
        self.db = lancedb.connect(db_path)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        logger.info(f"📊 Vector DB initialized at {db_path}")

    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def add_documents(self, documents: List[Document], session_id: str):
        """Add documents to vector DB with chunking"""
        logger.info(f"📝 Processing {len(documents)} documents...")

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.content)
            logger.info(f"   ↳ {doc.title}: {len(chunks)} chunks ({len(doc.content)} chars)")

            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "title": doc.title,
                    "source": doc.source,
                    "url": doc.url,
                    "doc_type": doc.doc_type,
                    "chunk_index": idx,
                    "session_id": session_id
                })

        # Embed all chunks
        logger.info(f"🔢 Embedding {len(all_chunks)} chunks...")
        texts = [c["text"] for c in all_chunks]
        embeddings = self.encoder.encode(texts, batch_size=128, show_progress_bar=False, device='cuda')

        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["vector"] = embedding.tolist()

        # Create or overwrite table for this session
        table_name = f"kb_{session_id}"
        try:
            self.db.drop_table(table_name)
        except:
            pass

        table = self.db.create_table(table_name, all_chunks)
        logger.info(f"✅ Stored {len(all_chunks)} chunks in {table_name}")
        return table

    def retrieve(self, query: str, session_id: str, top_k: int = TOP_K_CHUNKS) -> str:
        """Retrieve top-K most relevant chunks for query"""
        table_name = f"kb_{session_id}"

        try:
            table = self.db.open_table(table_name)
        except:
            logger.error(f"❌ Table {table_name} not found")
            return ""

        # Embed query
        query_embedding = self.encoder.encode([query], device='cuda')[0]

        # Vector search
        results = table.search(query_embedding.tolist()).limit(top_k).to_list()

        # Deduplicate and reconstruct context
        seen_chunks = set()
        context_parts = []

        logger.info(f"🔍 Retrieved {len(results)} relevant chunks:")
        for r in results:
            chunk_id = (r['title'], r['chunk_index'])
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                context_parts.append(f"--- {r['doc_type'].upper()}: {r['title']} ---\n{r['text']}\nSource: {r['url']}\n")
                logger.info(f"   ↳ {r['doc_type']}: {r['title']} (chunk {r['chunk_index']})")

        return "\n\n".join(context_parts)

# --- SEARCH ANALYSIS ---

def download_image(img_url: str, session_id: str, cache: dict) -> Optional[str]:
    """
    Download an image with deduplication, size filtering, and session organization.

    Args:
        img_url: URL of the image to download
        session_id: Current session ID for organizing images
        cache: Global cache dict (hash -> path mapping)

    Returns:
        Local file path if successful, None otherwise
    """
    try:
        # Make URL absolute if relative
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        elif img_url.startswith('/'):
            return None  # Skip relative URLs without base

        # Skip data URLs and SVGs (not useful for vision models)
        if img_url.startswith('data:') or img_url.endswith('.svg'):
            return None

        # Priority 2.1: URL pattern filtering (skip logos, icons, etc.)
        img_url_lower = img_url.lower()
        if any(pattern in img_url_lower for pattern in SKIP_IMAGE_PATTERNS):
            logger.debug(f"Skipping non-content URL pattern: {img_url}")
            return None

        # Download image data
        response = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            return None

        image_data = response.content

        # Priority 1.1: Content-based deduplication
        img_hash = get_image_hash(image_data)

        if img_hash in cache:
            # Image already exists from a previous session
            existing_path = cache[img_hash]
            if Path(existing_path).exists():
                logger.debug(f"Image already cached: {existing_path}")
                return existing_path

        # Priority 1.2: Size filtering (skip tiny images - logos, icons)
        try:
            img = Image.open(BytesIO(image_data))
            width, height = img.size

            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                logger.debug(f"Skipping small image ({width}x{height}): {img_url}")
                return None

            # Get proper file extension
            ext = img.format.lower() if img.format else 'jpg'
        except Exception as e:
            logger.debug(f"Failed to process image {img_url}: {e}")
            return None

        # Priority 1.3: Session-based organization
        session_dir = IMAGE_DIR / session_id / "images"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Use content hash as filename (automatic deduplication)
        filepath = session_dir / f"{img_hash}.{ext}"

        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_data)

        # Update cache
        cache[img_hash] = str(filepath)

        return str(filepath)

    except Exception as e:
        logger.debug(f"Failed to download image {img_url}: {e}")
        return None


def needs_visualization(query: str) -> bool:
    """Detect if query would benefit from visual diagrams"""
    lower = query.lower()
    return any(keyword in lower for keyword in VISUAL_KEYWORDS)


def get_search_info(user_prompt: str) -> dict:
    """Analyzes prompt using keyword matching"""
    print("\n=== SEARCH ANALYSIS ===")
    lower = user_prompt.lower()

    # ArXiv trigger - expanded keywords
    research_keywords = [
        "paper", "research", "study", "experiment", "trial", "clinical",
        "model", "algorithm", "theory", "physics", "biology", "chemistry",
        "arxiv", "science", "scientific", "publication", "journal"
    ]
    arxiv_search = any(k in lower for k in research_keywords)

    # Use full prompt for arXiv search
    arxiv_query = user_prompt if arxiv_search else None
    if arxiv_search:
        print(f"🔬 arXiv: ACTIVE - Query: '{arxiv_query}'")
    else:
        print("🔬 arXiv: Skipped")

    # Extract better search query - remove question words, keep key terms
    # This helps avoid getting wrong Wikipedia articles
    clean_web_query = re.sub(r'\b(describe|detail|explain|name|naming|tell me|what|how|why|when|where|which|who)\b', ' ', lower)
    clean_web_query = ' '.join(clean_web_query.split())  # Remove extra spaces
    web_search_query = clean_web_query if len(clean_web_query) > 10 else user_prompt

    print(f"📖 Wikipedia: Will search via web (auto-detect)")
    print(f"🔍 Cleaned search query: '{web_search_query}'")

    print("=== END ANALYSIS ===\n")

    return {
        "wiki_titles": [],
        "arxiv_search": arxiv_search,
        "arxiv_query": arxiv_query,
        "web_query": web_search_query  # Use cleaned query
    }

# --- DATA FETCHING ---

def fetch_wikipedia_full(query: str) -> List[Document]:
    """Fetch FULL Wikipedia articles (no sentence limits)"""
    documents = []
    print(f"--- WIKIPEDIA SEARCH ---")

    try:
        # Extract key terms for better Wikipedia matching
        # Remove common question words and extract noun phrases
        clean_query = re.sub(r'\b(describe|detail|explain|what|how|why|is|are|the|a|an|of|in|for|to|and)\b', ' ', query.lower())
        clean_query = ' '.join(clean_query.split())  # Remove extra spaces

        # Truncate to Wikipedia's 300 char limit (extract first key sentence)
        if len(clean_query) > 250:
            # Take first sentence or first 250 chars
            first_sentence = clean_query.split('.')[0]
            clean_query = first_sentence[:250] if len(first_sentence) < 250 else clean_query[:250]

        # Use cleaned query for Wikipedia search
        search_query = clean_query if len(clean_query) > 10 else query[:250]
        search_results = wikipedia.search(search_query, results=10)
        print(f"   ↳ Searching for: '{search_query}'")
        print(f"   ↳ Found {len(search_results)} potential articles")

        for title in search_results[:10]:  # Get top 10
            try:
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content  # FULL article content

                documents.append(Document(
                    title=page.title,
                    content=content,
                    source="Wikipedia",
                    url=page.url,
                    doc_type="wikipedia"
                ))
                print(f"   ✅ {page.title} ({len(content)} chars)")
            except Exception as e:
                print(f"   ⚠️  {title}: {e}")
                continue

    except Exception as e:
        print(f"   ❌ Wikipedia error: {e}")

    return documents

def fetch_arxiv_full(query: str, max_results: int = MAX_ARXIV_PAPERS) -> List[Document]:
    """Fetch FULL arXiv abstracts (no truncation)"""
    documents = []
    print(f"--- ARXIV SEARCH ---")

    try:
        encoded = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results={max_results}&sortBy=relevance"
        feed = feedparser.parse(url)

        print(f"   ↳ Found {len(feed.entries)} papers")

        for entry in feed.entries:
            # Get FULL abstract (no truncation)
            full_abstract = entry.summary

            documents.append(Document(
                title=entry.title,
                content=full_abstract,
                source="arXiv",
                url=entry.link,
                doc_type="arxiv"
            ))
            print(f"   ✅ {entry.title[:60]}... ({len(full_abstract)} chars)")

    except Exception as e:
        print(f"   ❌ arXiv error: {e}")

    return documents

def fetch_web_full(query: str, session_id: str, max_scrapes: int = MAX_WEB_SCRAPES) -> Tuple[List[Document], List[str]]:
    """
    Fetch and scrape FULL web pages AND download images.

    Args:
        query: Search query
        session_id: Session ID for organizing images
        max_scrapes: Maximum number of pages to scrape

    Returns:
        (text_documents, image_paths)
    """
    documents = []
    image_paths = []
    image_metadata = []  # Track metadata for session

    # Load global image cache for deduplication
    cache = load_image_cache()

    print(f"--- WEB SEARCH & SCRAPE ---")

    try:
        # Get search results
        results = DDGS().text(query, max_results=max_scrapes)

        if not results:
            print("   ❌ No web results found")
            return documents, image_paths

        print(f"   ↳ Found {len(results)} URLs, scraping content...")

        for idx, r in enumerate(results[:max_scrapes], 1):
            try:
                # Check if it's a PDF
                is_pdf = r['href'].lower().endswith('.pdf') or '[PDF]' in r.get('title', '')

                if is_pdf:
                    # Try to extract text from PDF
                    print(f"   📄 [{idx}] [PDF] {r['title'][:50]}... (extracting)")
                    content = extract_pdf_text(r['href'], timeout=10)

                    if content and len(content) > 200:
                        documents.append(Document(
                            title=r['title'],
                            content=content,
                            source="PDF",
                            url=r['href'],
                            doc_type="pdf"
                        ))
                        print(f"   ✅ [{idx}] [PDF] {r['title'][:50]}... ({len(content)} chars)")
                    else:
                        print(f"   ⚠️  [{idx}] [PDF] {r['title'][:50]}... (extraction failed)")
                    continue

                # Regular HTML scraping
                try:
                    response = requests.get(r['href'], timeout=5, headers={'User-Agent': 'Mozilla/5.0'})

                    # Check content-type to catch PDFs not ending in .pdf
                    content_type = response.headers.get('content-type', '').lower()
                    if 'application/pdf' in content_type:
                        print(f"   📄 [{idx}] [PDF] {r['title'][:50]}... (extracting)")
                        content = extract_pdf_text(r['href'], timeout=10)

                        if content and len(content) > 200:
                            documents.append(Document(
                                title=r['title'],
                                content=content,
                                source="PDF",
                                url=r['href'],
                                doc_type="pdf"
                            ))
                            print(f"   ✅ [{idx}] [PDF] {r['title'][:50]}... ({len(content)} chars)")
                        else:
                            print(f"   ⚠️  [{idx}] [PDF] {r['title'][:50]}... (extraction failed)")
                        continue

                    downloaded = response.text if response.status_code == 200 else None
                except (requests.Timeout, requests.RequestException) as e:
                    print(f"   ⚠️  [{idx}] {r['title'][:50]}... (timeout/error)")
                    continue

                if downloaded:
                    # Extract main content (removes ads, nav, etc.)
                    content = trafilatura.extract(downloaded, include_comments=False)

                    if content and len(content) > 200:  # Only keep substantial content
                        documents.append(Document(
                            title=r['title'],
                            content=content,
                            source="Web",
                            url=r['href'],
                            doc_type="web"
                        ))
                        print(f"   ✅ [{idx}] {r['title'][:50]}... ({len(content)} chars)")

                        # Extract images from the page
                        try:
                            soup = BeautifulSoup(downloaded, 'html.parser')
                            img_tags = soup.find_all('img', src=True)

                            # Download images with deduplication and filtering
                            downloaded_count = 0
                            skipped_count = 0

                            for img in img_tags:
                                if downloaded_count >= 5:  # Max 5 images per page
                                    break

                                img_url = img['src']
                                img_path = download_image(img_url, session_id, cache)

                                if img_path:
                                    image_paths.append(img_path)
                                    downloaded_count += 1

                                    # Collect metadata
                                    image_metadata.append({
                                        "hash": Path(img_path).stem,
                                        "filename": Path(img_path).name,
                                        "source_url": img_url,
                                        "page_title": r['title'],
                                        "page_url": r['href'],
                                        "alt_text": img.get('alt', '')
                                    })
                                else:
                                    skipped_count += 1

                            if downloaded_count > 0:
                                print(f"      ↳ Downloaded {downloaded_count} images (skipped {skipped_count})")

                        except Exception as img_error:
                            logger.debug(f"Image extraction failed for {r['href']}: {img_error}")

                    else:
                        print(f"   ⚠️  [{idx}] {r['title'][:50]}... (content too short)")
                else:
                    print(f"   ⚠️  [{idx}] {r['title'][:50]}... (fetch failed)")

            except Exception as e:
                print(f"   ⚠️  [{idx}] Error scraping: {e}")
                continue

    except Exception as e:
        print(f"   ❌ Web search error: {e}")

    # Save updated cache
    save_image_cache(cache)

    # Save session metadata
    if image_metadata:
        save_session_metadata(session_id, query, image_metadata)

    return documents, image_paths

def extract_pdf_text(pdf_url: str, timeout: int = 10) -> Optional[str]:
    """
    Extract text from a PDF URL.
    Returns extracted text or None if extraction fails.
    """
    try:
        response = requests.get(pdf_url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code != 200:
            return None

        # Parse PDF from bytes
        pdf_file = BytesIO(response.content)
        reader = PdfReader(pdf_file)

        # Extract text from all pages
        text_chunks = []
        for page in reader.pages[:20]:  # Limit to first 20 pages for performance
            text = page.extract_text()
            if text:
                text_chunks.append(text)

        full_text = '\n\n'.join(text_chunks)

        # Clean up common PDF artifacts
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
        full_text = re.sub(r'(\w)-\s+(\w)', r'\1\2', full_text)  # Fix hyphenation

        return full_text if len(full_text) > 200 else None

    except Exception as e:
        logger.debug(f"PDF extraction failed: {e}")
        return None


def convert_webp_to_png(image_path: str) -> str:
    """
    Convert .webp images to .png for LLaVA compatibility.
    Returns the path to the converted image (or original if no conversion needed).
    """
    path = Path(image_path)

    # Only convert .webp files
    if path.suffix.lower() != '.webp':
        return image_path

    try:
        # Create .png path (same location, different extension)
        png_path = path.with_suffix('.png')

        # Skip if already converted
        if png_path.exists():
            return str(png_path)

        # Convert webp -> png
        img = Image.open(image_path)
        img.save(png_path, 'PNG')

        print(f"   🔄 Converted {path.name} → {png_path.name}")
        return str(png_path)

    except Exception as e:
        logger.warning(f"Failed to convert {image_path}: {e}")
        return image_path  # Return original if conversion fails

def understand_images_with_llava(image_paths: List[str]) -> List[Document]:
    """
    Use LLaVA 13B to understand images and generate detailed descriptions.
    Returns: List of Document objects containing image descriptions
    """
    if not image_paths:
        return []

    documents = []
    print(f"\n--- IMAGE UNDERSTANDING (LLaVA 13B) ---")
    print(f"   ↳ Processing {len(image_paths)} images...")

    for idx, img_path in enumerate(image_paths, 1):
        try:
            # Convert .webp to .png if needed (LLaVA compatibility)
            processed_path = convert_webp_to_png(img_path)

            # Use LLaVA to understand the image
            response = ollama.chat(
                model="llava:13b",
                messages=[{
                    'role': 'user',
                    'content': '''Analyze this image in detail. Describe:
                    1. What the image shows (diagrams, charts, photos, illustrations)
                    2. Key visual elements, labels, and annotations
                    3. Scientific/technical content if present (molecules, circuits, graphs, etc.)
                    4. Text content visible in the image
                    5. Relationships between elements

                    Be comprehensive and technical.''',
                    'images': [processed_path]
                }]
            )

            description = response['message']['content']

            if description and len(description) > 50:
                # Extract filename for title
                filename = Path(img_path).name

                documents.append(Document(
                    title=f"Image: {filename}",
                    content=description,
                    source="Vision (LLaVA)",
                    url=img_path,
                    doc_type="image"
                ))
                print(f"   ✅ [{idx}] {filename} ({len(description)} chars)")
            else:
                print(f"   ⚠️  [{idx}] {Path(img_path).name} (description too short)")

        except Exception as e:
            logger.warning(f"Failed to process image {img_path}: {e}")
            print(f"   ⚠️  [{idx}] {Path(img_path).name} (processing failed)")
            continue

    return documents

# --- MAIN LOOP ---

def main():
    print("\n⚡ Resonance Governor with LanceDB Vector Retrieval ⚡")
    print("Features: Full Wikipedia | Full arXiv | Full Web Scraping | Vector Search | Vision (LLaVA)")

    # Initialize vector DB
    kb = VectorKnowledgeBase()
    session_counter = 0

    while True:
        prompt = input("\nUser: ").strip()
        if prompt.lower() in ['q', 'exit', 'quit']:
            break
        if not prompt:
            continue

        session_counter += 1
        session_id = f"session_{session_counter}"

        # 1. Analyze query
        info = get_search_info(prompt)

        # 2. Fetch all documents
        all_documents = []
        all_image_paths = []

        # Wikipedia
        all_documents.extend(fetch_wikipedia_full(info['web_query']))

        # arXiv
        if info['arxiv_search']:
            all_documents.extend(fetch_arxiv_full(info['arxiv_query']))

        # Web scraping (returns both text documents and image paths)
        web_docs, web_images = fetch_web_full(info['web_query'], session_id)
        all_documents.extend(web_docs)
        all_image_paths.extend(web_images)

        # Image understanding with LLaVA (if images were collected)
        if all_image_paths:
            # Deduplicate image paths (same image may be linked from multiple pages)
            unique_image_paths = list(dict.fromkeys(all_image_paths))
            skipped_dupes = len(all_image_paths) - len(unique_image_paths)

            if skipped_dupes > 0:
                print(f"\n📸 Found {len(all_image_paths)} image references ({skipped_dupes} duplicates), analyzing {len(unique_image_paths)} unique images with LLaVA...")
            else:
                print(f"\n📸 Found {len(unique_image_paths)} unique images, analyzing with LLaVA...")

            image_docs = understand_images_with_llava(unique_image_paths)
            all_documents.extend(image_docs)

        if not all_documents:
            print("❌ No documents retrieved. Try a different query.")
            continue

        # 3. Add to vector DB
        total_chars = sum(len(doc.content) for doc in all_documents)
        print(f"\n📊 Total collected: {len(all_documents)} documents, {total_chars:,} chars")

        kb.add_documents(all_documents, session_id)

        # 4. Vector retrieval
        print(f"\n🎯 Retrieving top {TOP_K_CHUNKS} relevant chunks for query...")
        retrieved_context = kb.retrieve(prompt, session_id, top_k=TOP_K_CHUNKS)

        if len(retrieved_context) < 100:
            print("❌ Insufficient context retrieved.")
            continue

        print(f"\n✅ Anchor Locked ({len(retrieved_context):,} chars). Governor Active.")

        # 5. Detect if visualization diagrams would be helpful
        use_visualization = needs_visualization(prompt)

        if use_visualization:
            print("🎨 Diagram mode: Will generate Python visualization code (Llama3)")
        else:
            print("📝 Text mode: Standard response (Llama3)")

        # 6. Run Governor (always use Llama3 for text generation)
        try:
            wrapper = OllamaResonanceWrapper(
                retrieved_context,
                model="llama3",
                enable_diagrams=use_visualization
            )
            print("\n--- 🟢 RESPONSE ---")
            for chunk in wrapper.stream_chat(prompt):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ Governor Error: {e}")

if __name__ == "__main__":
    main()
