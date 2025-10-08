import os
import sys
import logging
import shutil
import uuid
import datetime
from typing import List, Dict, Any, Optional

# Vector database
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from together import Together

# Add this after all the imports and before the SimpleTextLoader class
class SimpleCustomRetriever:
    """A simplified custom retriever implementation that works with the RetrievalQA chain."""
    
    def __init__(self, vectorstore):
        """Initialize with a vector store."""
        self.vectorstore = vectorstore
        
    def get_relevant_documents(self, query):
        """Get documents relevant to a query."""
        try:
            # Use the vectorstore's similarity search
            if hasattr(self.vectorstore, 'similarity_search'):
                docs = self.vectorstore.similarity_search(query, k=6)
                return docs
            else:
                # Fallback to no documents if similarity_search doesn't exist
                logger.warning("VectorStore has no similarity_search method")
                return []
        except Exception as e:
            logger.error(f"Error in get_relevant_documents: {e}")
            return []
            
    # This is the method langchain looks for in newer versions
    def _get_relevant_documents(self, query):
        """Compatibility method for newer langchain versions."""
        return self.get_relevant_documents(query)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimpleTextLoader:
    """Load text documents."""
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        """Load from a text file."""
        from langchain.docstore.document import Document
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

class SimpleCSVLoader:
    """Load CSV documents."""
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        """Load from a CSV file."""
        import csv
        from langchain.docstore.document import Document
        
        docs = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            for i, row in enumerate(csv_reader):
                # Convert row to a dictionary
                content = {headers[j]: value for j, value in enumerate(row) if j < len(headers)}
                # Convert to string
                content_str = "\n".join([f"{k}: {v}" for k, v in content.items()])
                metadata = {"source": self.file_path, "row": i}
                docs.append(Document(page_content=content_str, metadata=metadata))
        return docs

class SimplePDFLoader:
    """Load PDF documents."""
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        """Load from a PDF file."""
        import pypdf
        from langchain.docstore.document import Document
        
        docs = []
        with open(self.file_path, "rb") as file:
            pdf = pypdf.PdfReader(file)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                metadata = {"source": self.file_path, "page": i}
                docs.append(Document(page_content=text, metadata=metadata))
        return docs

class TogetherLLM(LLM):
    """Custom LangChain LLM wrapper for Together's API."""
    
    client: Any  # Together client
    model_name: str
    temperature: float = 0.1
    max_tokens: int = 1024
    
    def _call(self, prompt: str, **kwargs) -> str:
        """Call the Together API."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Together API: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

class SimpleLegalRAG:
    """
    A simplified RAG system for legal support.
    """
    def __init__(
        self,
        docs_dir: str = "legal_documents",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "legal_vectordb",
        together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        together_api_key: Optional[str] = None,
    ):
        self.docs_dir = docs_dir
        self.embeddings_model = embeddings_model
        self.persist_dir = persist_dir
        self.together_model = together_model
        self.together_api_key = together_api_key or os.environ.get("TOGETHER_API_KEY")
        
        # Initialize components
        self.embeddings = None
        self.vectordb = None
        self.llm = None
        self.qa_chain = None
        
        # Track document modifications to know when to rebuild
        self.last_document_hash = None
        
        # Create directories if they don't exist
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"Initialized Legal RAG with docs directory: {docs_dir}")
    
    def _get_documents_hash(self):
        """Generate a hash of the current state of documents for change tracking."""
        doc_info = []
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(('.pdf', '.txt', '.csv')):
                    full_path = os.path.join(root, file)
                    mtime = os.path.getmtime(full_path)
                    size = os.path.getsize(full_path)
                    doc_info.append(f"{full_path}:{mtime}:{size}")
        
        doc_info.sort()  # Sort for consistent hash
        return ",".join(doc_info)
    
    def setup(self):
        """Initialize the system."""
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        logger.info("Embeddings model initialized")
        
        # Initialize LLM
        together_client = Together(api_key=self.together_api_key)
        self.llm = TogetherLLM(
            client=together_client,
            model_name=self.together_model,
            temperature=0.0,
            max_tokens=2048
        )
        logger.info("LLM initialized")
        
        # Get current document hash
        current_doc_hash = self._get_documents_hash()
        self.last_document_hash = current_doc_hash
        
        # Initialize vector database
        need_rebuild = True
        if os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
            try:
                # Try to load existing database
                self.vectordb = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector database")
                
                # Only skip rebuild if there are documents in the vectordb
                try:
                    count = self.vectordb._collection.count()
                    if count > 0:
                        need_rebuild = False
                        logger.info(f"Vector database contains {count} entries")
                except:
                    logger.warning("Could not check vector database entries")
                    
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
                self.vectordb = None  # Will be recreated below
        
        # Rebuild if needed
        if need_rebuild:
            logger.info("Need to (re)build vector database")
            self.process_all_documents(force=True)
        else:
            # Still create QA chain
            self.create_qa_chain()
    
    def load_documents(self):
        """Load all documents from the docs directory with better metadata preservation."""
        logger.info(f"Loading documents from {self.docs_dir}")
        
        documents = []
        
        # Walk through all files in the directory
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Only process supported file types
                if file.endswith(".pdf"):
                    try:
                        loader = SimplePDFLoader(file_path)
                        file_docs = loader.load()
                        # Add better metadata
                        for doc in file_docs:
                            doc.metadata["file_name"] = file
                            doc.metadata["document_name"] = os.path.splitext(file)[0]
                            doc.metadata["document_type"] = "pdf"
                        documents.extend(file_docs)
                        logger.info(f"Loaded PDF: {file}")
                    except Exception as e:
                        logger.error(f"Error loading PDF {file}: {e}")
                
                elif file.endswith(".txt"):
                    try:
                        loader = SimpleTextLoader(file_path)
                        file_docs = loader.load()
                        # Add better metadata
                        for doc in file_docs:
                            doc.metadata["file_name"] = file
                            doc.metadata["document_name"] = os.path.splitext(file)[0]
                            doc.metadata["document_type"] = "txt"
                        documents.extend(file_docs)
                        logger.info(f"Loaded TXT: {file}")
                    except Exception as e:
                        logger.error(f"Error loading TXT {file}: {e}")
                
                elif file.endswith(".csv"):
                    try:
                        loader = SimpleCSVLoader(file_path)
                        file_docs = loader.load()
                        # Add better metadata
                        for doc in file_docs:
                            doc.metadata["file_name"] = file
                            doc.metadata["document_name"] = os.path.splitext(file)[0]
                            doc.metadata["document_type"] = "csv"
                        documents.extend(file_docs)
                        logger.info(f"Loaded CSV: {file}")
                    except Exception as e:
                        logger.error(f"Error loading CSV {file}: {e}")
        
        # Log the loaded documents
        logger.info(f"Loaded {len(documents)} total documents")
        for doc in documents[:5]:  # Log just the first 5 to avoid excessive logging
            logger.info(f"Document: {doc.metadata.get('document_name')} ({doc.metadata.get('file_name')})")
        
        return documents
    
    def process_all_documents(self, force=False):
        """Process all documents and update the vector database with improved metadata."""
        # Load all documents
        documents = self.load_documents()
        
        if not documents:
            logger.warning("No documents to process")
            # Make sure we have a QA chain even with no documents
            self.create_qa_chain()
            return
        
        logger.info(f"Processing {len(documents)} documents")
        
        # Split documents into chunks with better metadata preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for better precision
            chunk_overlap=100,  # More overlap to ensure context is preserved
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Enhance chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            # Ensure document_name is preserved
            if "document_name" not in chunk.metadata and "file_name" in chunk.metadata:
                chunk.metadata["document_name"] = os.path.splitext(chunk.metadata["file_name"])[0]
        
        # Log details about the chunks
        for i, chunk in enumerate(chunks[:min(3, len(chunks))]):
            logger.info(f"Sample chunk {i}: {chunk.page_content[:100]}... from {chunk.metadata.get('document_name', 'Unknown')}")
        
        logger.info(f"Split into {len(chunks)} chunks")
        
        try:
            # Close existing database connection
            if self.vectordb:
                try:
                    self.vectordb._collection = None
                    self.vectordb = None
                except:
                    pass
            
            # Clear files in vector database directory without removing the directory
            try:
                for item in os.listdir(self.persist_dir):
                    item_path = os.path.join(self.persist_dir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
            except Exception as e:
                logger.warning(f"Could not clear vector database files: {e}")
            
            # Create new vector database with improved settings for better retrieval
            self.vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            # Explicitly persist the vector database
            self.vectordb.persist()
            logger.info(f"Created new vector database with {len(chunks)} chunks")
            logger.info("Vector database persisted to disk")
        except Exception as e:
            logger.error(f"Error creating vector database: {e}")
            # Try to create an empty database as fallback
            try:
                self.vectordb = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                logger.warning("Fallback: Using empty vector database")
            except Exception as e2:
                logger.error(f"Failed to create empty DB: {e2}")
        
        # Create QA chain
        self.create_qa_chain()
    
    def create_qa_chain(self):
        """Create the QA chain with a better prompt that emphasizes document specificity."""
        logger.info("Creating QA chain")
        
        # Create a more detailed prompt template that emphasizes document specificity
        prompt_template = """
        You are a legal assistant that ONLY provides information found in the provided context.
        If the answer cannot be found in the context, respond with:
        "I don't have enough information in the provided documents to answer this question. Please upload relevant documents or try a different question."
        
        Context:
        {context}
        
        Question: {question}
        
        VERY IMPORTANT INSTRUCTIONS:
        1. Pay special attention to which document the user is asking about. If they mention a specific document name like "Retailer Agreement_4" in their question, ONLY provide information from that specific document.
        2. Always identify which document contains your answer by name (e.g., "According to Retailer Agreement_4...")
        3. The context contains metadata for each chunk, including 'document_name' which tells you which document it comes from.
        4. Be precise and focus on answering exactly what was asked.
        5. If asked about a specific section (like "point 6"), look for that exact section number in the documents.
        6. Use ONLY information from the context provided. Do not use prior knowledge.
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        if self.vectordb is None:
            logger.error("Vector database is None when creating QA chain")
            # Create empty vector database
            self.vectordb = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        
        # Create our custom retriever
        try:
            # Import our custom retriever class
            # This is defined at the top of the file
            retriever = SimpleCustomRetriever(self.vectordb)
            logger.info("Created custom retriever")
            
            # Define a simple chain that doesn't rely on RetrievalQA
            from langchain.chains import LLMChain
            
            def get_context(query):
                """Get context for a query."""
                try:
                    docs = retriever.get_relevant_documents(query)
                    return "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    logger.error(f"Error getting context: {e}")
                    return ""
            
            # Create a simple chain that works around RetrievalQA issues
            def qa_chain_function(query_dict):
                """A function that mimics the behavior of RetrievalQA."""
                query = query_dict.get("query", "")
                try:
                    # Get context for the query
                    context = get_context(query)
                    
                    # If no context, return a default response
                    if not context:
                        return {
                            "result": "I don't have enough information in the provided documents to answer this question. Please upload relevant documents or try a different question.",
                            "source_documents": []
                        }
                    
                    # Format prompt with context and query
                    formatted_prompt = PROMPT.format(context=context, question=query)
                    
                    # Call LLM
                    response = self.llm(formatted_prompt)
                    
                    # Return response in the format expected by the API
                    return {
                        "result": response,
                        "source_documents": retriever.get_relevant_documents(query)
                    }
                except Exception as e:
                    logger.error(f"Error in qa_chain_function: {e}")
                    return {
                        "result": "An error occurred while processing your query. Please try again.",
                        "source_documents": []
                    }
            
            # Set the qa_chain to our function
            self.qa_chain = qa_chain_function
            logger.info("Created custom QA chain function")
            
        except Exception as e:
            logger.error(f"Error creating custom QA chain: {e}")
            # Create a simple fallback function
            def simple_fallback(query_dict):
                return {
                    "result": "The system is currently experiencing technical difficulties. Please try again later or contact support.",
                    "source_documents": []
                }
            self.qa_chain = simple_fallback
            logger.error("Using simple fallback QA function")
    
    def query(self, question):
        """Answer a question using the RAG system with direct document access."""
        logger.info(f"Received query: {question}")
        
        # Extract document references from the question
        doc_names = []
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                doc_name = os.path.splitext(file)[0]
                if doc_name.lower() in question.lower():
                    doc_names.append(doc_name)
                    logger.info(f"Question references document: {doc_name}")
        
        # Check if we have documents
        documents = []
        try:
            # Load all documents directly rather than relying on the vector database
            documents = self.load_documents()
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
        
        if not documents:
            logger.info("No documents in system")
            return {
                "question": question,
                "answer": "There are no documents in the system yet. Please upload legal documents to get answers to your questions.",
                "sources": []
            }
        
        # If qa_chain exists, try to use it first
        if self.qa_chain is not None:
            try:
                # Try using the vector retriever first
                if doc_names:
                    modified_question = f"Referring specifically to {', '.join(doc_names)}: {question}"
                else:
                    modified_question = question
                    
                result = self.qa_chain({"query": modified_question})
                
                # If we got meaningful results, return them
                if result and result.get("result") and "I don't have enough information" not in result.get("result"):
                    # Process sources
                    sources = []
                    if "source_documents" in result:
                        for doc in result["source_documents"]:
                            source = {
                                "content": doc.page_content[:200] + "...",
                                "source": doc.metadata.get('document_name', doc.metadata.get('source', 'Unknown'))
                            }
                            sources.append(source)
                    
                    return {
                        "question": question,
                        "answer": result.get("result", ""),
                        "sources": sources
                    }
            except Exception as e:
                logger.error(f"Error in qa_chain retrieval: {e}")
        
        # If vector retrieval failed or returned no results, use direct document search as fallback
        logger.info("Using direct document search as fallback")
        
        # Filter documents if specific document is mentioned
        if doc_names:
            filtered_docs = []
            for doc in documents:
                doc_name = doc.metadata.get('document_name', '')
                if doc_name in doc_names:
                    filtered_docs.append(doc)
            
            if filtered_docs:
                documents = filtered_docs
                logger.info(f"Filtered to {len(documents)} documents matching {doc_names}")
        
        # Create context from documents by simple keyword matching
        relevant_docs = []
        question_words = set(question.lower().split())
        
        # Remove common words to focus on important keywords
        stop_words = {'what', 'where', 'when', 'who', 'how', 'why', 'is', 'are', 'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'with', 'about', 'tell', 'me'}
        question_words = question_words - stop_words
        
        # Simple relevance scoring
        for doc in documents:
            doc_content = doc.page_content.lower()
            score = sum(1 for word in question_words if word in doc_content)
            if score > 0:
                relevant_docs.append((doc, score))
        
        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 docs
        top_docs = [doc for doc, _ in relevant_docs[:5]]
        
        if not top_docs:
            # If still no relevant docs, just use the first few documents
            top_docs = documents[:5]
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in top_docs])
        
        # Create prompt
        prompt_template = """
        You are a legal assistant that ONLY provides information found in the provided context.
        If the answer cannot be found in the context, respond with:
        "I don't have enough information in the provided documents to answer this question. Please upload relevant documents or try a different question."
        
        Context:
        {context}
        
        Question: {question}
        
        VERY IMPORTANT INSTRUCTIONS:
        1. Pay special attention to which document the user is asking about. If they mention a specific document name like "Retailer Agreement_4" in their question, ONLY provide information from that specific document.
        2. Always identify which document contains your answer by name (e.g., "According to Retailer Agreement_4...")
        3. The context contains metadata for each chunk, including 'document_name' which tells you which document it comes from.
        4. Be precise and focus on answering exactly what was asked.
        5. If asked about a specific section (like "point 6"), look for that exact section number in the documents.
        6. Use ONLY information from the context provided. Do not use prior knowledge.
        """
        
        formatted_prompt = prompt_template.format(context=context, question=question)
        
        try:
            # Call LLM directly
            answer = self.llm(formatted_prompt)
            
            # Process sources for response
            sources = []
            for doc in top_docs:
                source = {
                    "content": doc.page_content[:200] + "...",
                    "source": doc.metadata.get('document_name', doc.metadata.get('source', 'Unknown'))
                }
                sources.append(source)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in direct document search: {e}")
            return {
                "question": question,
                "answer": "An error occurred while processing your question. Please try again or contact support.",
                "sources": []
            }
    
    def add_document(self, filepath):
        """Add a new document to the system."""
        logger.info(f"Adding document: {filepath}")
        
        # Force complete rebuild of the system
        self.process_all_documents(force=True)
        
        return True
    
    def remove_document(self, filename):
        """Remove a document from the system."""
        logger.info(f"Removing document: {filename}")
        
        file_path = os.path.join(self.docs_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return False
        
        # Remove the file
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            
            # Force complete rebuild
            self.process_all_documents(force=True)
            
            return True
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False
    
    def reset(self):
        """Reset the entire system."""
        logger.info("Resetting system")
        
        # Close vector database if it exists
        if self.vectordb:
            try:
                self.vectordb._collection = None
                self.vectordb = None
            except:
                pass
        
        # Clear files in vector database directory without removing the directory
        try:
            for item in os.listdir(self.persist_dir):
                item_path = os.path.join(self.persist_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                # Skip subdirectories
        except Exception as e:
            logger.error(f"Error clearing vector database: {e}")
        
        # Create fresh vector database
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        
        # Process documents
        self.process_all_documents(force=True)
        
        return True