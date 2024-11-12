from qdrant_client import QdrantClient
import requests
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Document, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from typing import List
import uuid

def open_file_picker():
    root = tk.Tk()
    root.withdraw()
    root.lift()      # Lift the file dialog above other windows
    root.attributes('-topmost', True)  # Keep it above other windows
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path


PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_PATH = PROJECT_ROOT / "local_data" / "private_gpt"


QDRANT_PATH = LOCAL_DATA_PATH / "qdrant"

class OllamaEmbedder:
    def __init__(self, model="mxbai-embed-large", api_base="http://localhost:11434"):
        self.model = model
        self.api_base = api_base.rstrip('/')
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.api_base}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            embedding = response.json()["embedding"]
            embeddings.append(embedding)
        
        return np.array(embeddings)

# Initialize storage context with the same structure as PrivateGPT
try:
    docstore = SimpleDocumentStore.from_persist_dir(persist_dir=LOCAL_DATA_PATH)
except FileNotFoundError:
    print("Local document store not found, creating a new one")
    docstore = SimpleDocumentStore()

try:
    index_store = SimpleIndexStore.from_persist_dir(persist_dir=LOCAL_DATA_PATH)
except FileNotFoundError:
    print("Local index store not found, creating a new one")
    index_store = SimpleIndexStore()

storage_context = StorageContext.from_defaults(
    docstore=docstore,
    index_store=index_store,
    vector_store=QdrantVectorStore(
        client=QdrantClient(path=str(QDRANT_PATH)),
        collection_name="make_this_parameterizable_per_api_call"
    )
)

# Needs to be the same as the PrivateGPT embedding model
embedder = OllamaEmbedder(model="mxbai-embed-large")

# Initialize node parser
node_parser = SentenceWindowNodeParser.from_defaults()

def chunk_text(text, chunk_context="", delimiter="-+-+-+-", add_chunk_numbers=False):
    """
    Custom text chunking function that splits on a delimiter and adds context to each chunk
    
    Args:
        text: The text to chunk
        chunk_context: Context string to add at the end of each chunk
        delimiter: The string to split on
        add_chunk_numbers: If True, adds "Chunk X of Y" to the context
    """
    # Split on delimiter
    chunks = text.split(delimiter)
    
    # Remove empty chunks and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Prepare the final chunks with context
    final_chunks = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk
        
        # Build context string
        context = []
        if add_chunk_numbers:
            context.append(f"[Chunk {i} of {total_chunks}]")
        if chunk_context:
            context.append(chunk_context)
        
        # Add context if provided
        if context:
            context_str = " ".join(context)
            chunk_text = f"{chunk_text}\n\n[Context: {context_str}]"
        
        final_chunks.append(chunk_text)
    
    return final_chunks

def create_document_nodes(chunks: List[str], embeddings: List[List[float]], file_name: str) -> List[Document]:
    """Create Document nodes from chunks with embeddings"""
    nodes = []
    for chunk, embedding in zip(chunks, embeddings):
        # Generate a unique doc_id
        doc_id = str(uuid.uuid4())
        
        # Create metadata dict with fields that match privategpt implementation
        doc_metadata = {
            "doc_id": doc_id,
            "file_name": file_name
        }
        
        # Create Document node with embedding and UUID
        doc = Document(
            text=chunk,
            metadata=doc_metadata,
            id_=doc_id
        )
        # Set the embedding
        doc.embedding = embedding
        
        # Set excluded metadata keys to match PrivateGPT
        doc.excluded_embed_metadata_keys = ["doc_id"]
        doc.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
        
        # Parse the document into nodes using the same parser as PrivateGPT
        nodes.extend(node_parser.get_nodes_from_documents([doc]))
    
    return nodes

def insert_document(text, delimiter="-+-+-+-", chunk_context=""):
    """
    Insert a document with custom chunking
    
    Args:
        text: The text to insert
        delimiter: The string to split chunks on
        chunk_context: Context string to add at the end of each chunk
    """
    # Get the file path from the file picker
    print("Select a file to process...")
    file_path = open_file_picker()
    
    if not file_path:
        print("No file selected. Exiting...")
        return
    
    # Get the file name
    file_name = Path(file_path).name
    print(f"\nProcessing file: {file_name}")
    
    # Custom chunk the text
    chunks = chunk_text(text, chunk_context=chunk_context, delimiter=delimiter)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings for chunks using Ollama
    print("Generating embeddings...")
    embeddings = embedder.encode(chunks)
    
    # Create Document nodes with embeddings
    print("Creating document nodes...")
    nodes = create_document_nodes(chunks, embeddings.tolist(), file_name)
    
    # Store documents in both vector store and doc store
    print("Adding to stores...")
    # Add to vector store
    storage_context.vector_store.add(nodes=nodes)
    # Add to doc store
    storage_context.docstore.add_documents(nodes)
    
    # Persist the storage context
    storage_context.persist(persist_dir=LOCAL_DATA_PATH)
    print("Done!")

if __name__ == "__main__":
    print("Select a file to process...")
    file_path = open_file_picker()
    
    if not file_path:
        print("No file selected. Exiting...")
        exit()
        
    print(f"\nReading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("\nProcessing document...")
    insert_document(
        text,
        delimiter="-+-+-+-",
        chunk_context=f"Source: {Path(file_path).name}"
    )
