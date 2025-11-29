"""
IntelliDocs AI - Azure AI Search PRIMARY Implementation
Institute Project Version (Patched for SemanticSettings + .env)
"""

import os
import json
import uuid
import sys
import logging 
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import DocumentProcessor from separate file
from document_processor import DocumentProcessor

# Check if processors are available
try:
    import PyPDF2
    from docx import Document as DocxDocument
    from pptx import Presentation
    import openpyxl
    from openpyxl import load_workbook
    PROCESSORS_AVAILABLE = True
    logger.info("✅ All document processors available")
except ImportError as e:
    PROCESSORS_AVAILABLE = False
    logger.warning(f"⚠️ Some processors missing: {e}")

# ---------------------------
# Load .env from project root
# ---------------------------
from dotenv import load_dotenv

# Determine project root: backend/main.py -> parents[2] reaches project root
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print(f"⚠️ .env not found at {env_path} — ensure your .env is at project root")

# Configure logging for Azure
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------
# Azure SDK imports (compat)
# ---------------------------
try:
    # Core Azure Search imports
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    # Try to import models that may vary across versions
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SimpleField,
        SearchableField,
        SearchFieldDataType,
        SearchField,
        VectorSearch,
        SemanticConfiguration,
        SemanticField,
        SearchSuggester,
    )

    # Try to import SemanticSettings — if missing, we'll create a compatibility shim below
    try:
        from azure.search.documents.indexes.models import SemanticSettings  # may not exist in newer SDKs
        SEMANTIC_SETTINGS_AVAILABLE = True
    except Exception:
        SemanticSettings = None
        SEMANTIC_SETTINGS_AVAILABLE = False

    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError

except Exception as e:
    # Provide a clear message with the recommended versions
    raise ImportError(
        "❌ Azure SDK import failed or version mismatch.\n"
        "Recommended fix (in your activated .venv):\n"
        "pip install 'azure-search-documents==11.4.0' 'azure-core==1.29.4' 'azure-common==1.1.28'\n"
        f"Original error: {e}"
    ) from e

# If SemanticSettings is not available in installed SDK, define a small compatibility class
if not SEMANTIC_SETTINGS_AVAILABLE:
    class SemanticSettings:
        """
        Compatibility shim for SemanticSettings if it's not present in the installed Azure SDK.
        This class only stores configurations and mirrors the expected interface minimally.
        """
        def __init__(self, configurations: Optional[List[Any]] = None):
            self.configurations = configurations or []

        def __repr__(self):
            return f"SemanticSettings(configurations={self.configurations!r})"

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Load environment variables (after dotenv)
# ---------------------------
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "employee-documents")
USE_AZURE_PRIMARY = os.getenv("USE_AZURE_PRIMARY", "true").lower() == "true"
FALLBACK_TO_LOCAL = os.getenv("FALLBACK_TO_LOCAL", "true").lower() == "true"

# Print environment values for debugging (non-sensitive)
print("🔍 Environment check:")
print("  AZURE_SEARCH_ENDPOINT =", AZURE_SEARCH_ENDPOINT)
print("  AZURE_SEARCH_INDEX_NAME =", INDEX_NAME)
print("  USE_AZURE_PRIMARY =", USE_AZURE_PRIMARY)
print("  FALLBACK_TO_LOCAL =", FALLBACK_TO_LOCAL)

# Validate Azure Configuration
if USE_AZURE_PRIMARY and (not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_ADMIN_KEY):
    raise ValueError(
        "❌ AZURE CREDENTIALS MISSING!\n"
        "Please update your .env file with:\n"
        "- AZURE_SEARCH_ENDPOINT\n"
        "- AZURE_SEARCH_ADMIN_KEY\n\n"
        "This project REQUIRES Azure AI Search as primary service.\n"
    )

# ---------------------------
# Local backup + upload dir
# ---------------------------
DOCUMENTS_DB_FILE = "documents_db_backup.json"
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(
    title="IntelliDocs AI - Azure AI Search Portal",
    description="Intelligent document search powered by Azure AI Search (PRIMARY)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Initialize Azure clients (PRIMARY)
# ---------------------------
index_client = None
search_client = None

if USE_AZURE_PRIMARY:
    try:
        admin_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        query_credential = AzureKeyCredential(AZURE_SEARCH_QUERY_KEY or AZURE_SEARCH_ADMIN_KEY)

        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=admin_credential
        )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            credential=query_credential
        )

        logger.info("✅ Azure AI Search clients initialized - PRIMARY MODE")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure AI Search: {e}")
        # If fallback is allowed, continue; otherwise raise
        if not FALLBACK_TO_LOCAL:
            raise

# ---------------------------
# Pydantic Models
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    top: Optional[int] = 10
    skip: Optional[int] = 0
    filter: Optional[str] = None
    orderby: Optional[str] = None
    search_mode: Optional[str] = "any"  # any or all
    query_type: Optional[str] = "simple"  # simple or full

class SearchResponse(BaseModel):
    count: int
    results: List[dict]
    facets: Optional[dict] = None
    suggestions: Optional[List[str]] = None

class IndexStats(BaseModel):
    document_count: int
    storage_size: int
    index_name: str
    status: str

# ---------------------------
# Index creation / update
# ---------------------------
async def create_or_update_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String,
                        sortable=True, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),
        SimpleField(name="category", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="file_size", type=SearchFieldDataType.Int64,
                    filterable=True, sortable=True),
        SimpleField(name="word_count", type=SearchFieldDataType.Int64,
                    filterable=True, sortable=True),
        SearchableField(name="tags", collection=True, type=SearchFieldDataType.String,
                        filterable=True, facetable=True)
    ]

    suggester = SearchSuggester(name="sg", source_fields=["title", "content"])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        suggesters=[suggester]
    )

    try:
        result = index_client.create_index(index)
        logger.info(f"✅ Created new index: {INDEX_NAME}")
        return result
    except Exception as e:
        if "already exists" in str(e).lower():
            result = index_client.create_or_update_index(index)
            logger.info(f"✅ Updated existing index: {INDEX_NAME}")
            return result
        else:
            logger.error(f"❌ Error creating index: {e}")
            raise

# ---------------------------
# API endpoints
# ---------------------------
@app.on_event("startup")
async def startup_event():
    try:
        if USE_AZURE_PRIMARY and index_client is not None:
            await create_or_update_index()
            logger.info("✅ Azure Search index ready")
    except Exception as e:
        logger.error(f"❌ Failed to setup index: {e}")

@app.get("/")
async def root():
    return {
        "message": "IntelliDocs AI - Azure AI Search Portal",
        "status": "running",
        "mode": "AZURE PRIMARY" if USE_AZURE_PRIMARY else "LOCAL",
        "azure_endpoint": AZURE_SEARCH_ENDPOINT,
        "index_name": INDEX_NAME,
        "version": "3.0.0",
        "endpoints": {
            "search": "/api/search",
            "upload": "/api/documents/upload-file",
            "index_stats": "/api/index/stats",
            "index_create": "/api/index/create",
            "suggestions": "/api/search/suggestions",
            "docs": "/docs"
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_documents_azure_primary(request: SearchRequest):
    try:
        logger.info(f"🔍 Azure Search Query: {request.query}")

        # Configure search options for Azure SDK
        search_options = {
            "search_text": request.query,
            "search_mode": request.search_mode,
            "include_total_count": True,
            "top": request.top,
            "skip": request.skip
        }

        if request.filter:
            search_options["filter"] = request.filter

        if request.orderby:
            search_options["order_by"] = request.orderby.split(",")

        azure_results = search_client.search(**search_options)

        results = []
        for result in azure_results:
            doc = {
                "id": result.get("id"),
                "title": result.get("title"),
                "content": result.get("content"),
                "category": result.get("category"),
                "created_date": result.get("created_date"),
                "score": result.get("@search.score", 0),
                "highlights": result.get("@search.highlights", {})
            }
            results.append(doc)

        count = azure_results.get_count() if hasattr(azure_results, 'get_count') else len(results)

        logger.info(f"✅ Azure Search returned {count} results")

        return SearchResponse(
            count=count,
            results=results
        )

    except Exception as azure_error:
        logger.error(f"❌ Azure Search failed: {azure_error}")
        if FALLBACK_TO_LOCAL:
            logger.info("⚠️ Falling back to local search")
            return await search_local_fallback(request)
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Azure Search unavailable: {str(azure_error)}"
            )

async def search_local_fallback(request: SearchRequest):
    try:
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            documents = []

        query_lower = request.query.lower()
        results = []

        for doc in documents:
            if query_lower in doc.get("title", "").lower() or \
               query_lower in doc.get("content", "").lower():
                results.append(doc)

        results = results[:request.top] if request.top else results

        return SearchResponse(
            count=len(results),
            results=results
        )
    except Exception as e:
        logger.error(f"Local search failed: {e}")
        return SearchResponse(count=0, results=[])

@app.post("/api/documents/upload-file")
async def upload_file_azure_primary(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_path = UPLOAD_DIR / file.filename
        content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Handle CSV manually
        if file.filename.lower().endswith(".csv"):
            import pandas as pd
            try:
                df = pd.read_csv(file_path)
                text_content = "\n".join(
                    df.astype(str).fillna("").apply(lambda row: " ".join(row), axis=1)
                )
                doc_data = {
                    "filename": file.filename,
                    "content": text_content,
                    "doc_type": "CSV",  # ← Changed to uppercase
                    "file_size": file_path.stat().st_size,
                    "word_count": len(text_content.split())
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")
        else:
            doc_data = DocumentProcessor.process_document(str(file_path))
            
        # Ensure consistent category naming
        category_map = {
            'pdf': 'PDF',
            'word': 'Word', 
            'docx': 'Word',
            'excel': 'Excel',
            'xlsx': 'Excel',
            'csv': 'CSV',
            'powerpoint': 'PowerPoint',
            'pptx': 'PowerPoint',
            'text': 'Text',
            'txt': 'Text'
        }
        
        doc_type = doc_data.get("doc_type", "unknown").lower()
        category = category_map.get(doc_type, doc_data["doc_type"])
        
        doc_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc).isoformat()

        azure_document = {
            "id": doc_id,
            "title": doc_data["filename"],
            "content": doc_data["content"][:32000],
            "category": category,  # ← Using mapped category
            "created_date": current_time,
            "file_size": doc_data.get("file_size", 0),
            "word_count": doc_data.get("word_count", 0),
            "tags": [category, "uploaded"]
        }

        # Continue with Azure upload...
        result = search_client.upload_documents(documents=[azure_document])
        if result[0].succeeded:
            logger.info(f"✅ Document uploaded to Azure: {doc_id}")
            if FALLBACK_TO_LOCAL:
                save_to_local_backup(azure_document)
            os.remove(file_path)
            return {
                "message": "Document indexed in Azure Search successfully",
                "document_id": doc_id,
                "filename": doc_data["filename"],
                "category": category,  # Return proper category
                "azure_indexed": True,
                "word_count": doc_data.get("word_count", 0)
            }
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/stats", response_model=IndexStats)
async def get_index_statistics():
    try:
        index = index_client.get_index(INDEX_NAME)
        count_result = search_client.get_document_count()

        return IndexStats(
            document_count=count_result,
            storage_size=0,
            index_name=INDEX_NAME,
            status="healthy"
        )
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/create")
async def create_index_endpoint():
    try:
        result = await create_or_update_index()
        return {
            "message": "Index created/updated successfully",
            "index_name": INDEX_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/suggestions")
async def get_search_suggestions(query: str, top: int = 5):
    try:
        suggestions = search_client.suggest(
            search_text=query,
            suggester_name="sg",
            top=top
        )
        return {
            "suggestions": [s["title"] for s in suggestions]
        }
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        return {"suggestions": []}

def save_to_local_backup(document: dict):
    try:
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, 'r', encoding='utf-8') as f:
                docs = json.load(f)
        else:
            docs = []

        docs.append(document)

        with open(DOCUMENTS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2)

        logger.info("📁 Document backed up locally")
    except Exception as e:
        logger.error(f"Local backup failed: {e}")

@app.get("/api/documents/recent")
async def get_recent_documents(limit: int = 10):
    """Get recently uploaded documents from Azure or local backup"""
    try:
        documents = []
        
        # Try Azure first
        if search_client:
            try:
                # Search for all documents, ordered by date
                results = search_client.search(
                    search_text="*",
                    order_by=["created_date desc"],
                    top=limit,
                    include_total_count=True
                )
                
                for doc in results:
                    documents.append({
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "category": doc.get("category"),
                        "created_date": doc.get("created_date"),
                        "content": doc.get("content", "")[:200],  # Preview
                        "preview": doc.get("content", "")[:100]
                    })
                    
            except Exception as e:
                logger.warning(f"Could not fetch from Azure: {e}")
        
        # Fallback to local if no Azure results
        if not documents and os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                local_docs = json.load(f)
            local_docs = sorted(local_docs, key=lambda x: x.get("created_date", ""), reverse=True)
            documents = local_docs[:limit]
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error fetching recent documents: {e}")
        return {"documents": [], "error": str(e)}
    
@app.delete("/api/documents/{document_id}")  # Fixed path
async def delete_document(document_id: str):
    """Delete a document from Azure Search and local backup"""
    try:
        deleted_from_azure = False
        deleted_from_local = False
        
        # Delete from Azure
        if search_client:
            try:
                search_client.delete_documents(documents=[{"id": document_id}])
                logger.info(f"✅ Deleted from Azure: {document_id}")
                deleted_from_azure = True
            except Exception as e:
                logger.warning(f"Azure delete issue: {e}")
        
        # Delete from local backup
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                docs = json.load(f)
            
            original_count = len(docs)
            docs = [d for d in docs if d.get("id") != document_id]
            
            if len(docs) < original_count:
                with open(DOCUMENTS_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2)
                deleted_from_local = True
                logger.info(f"✅ Deleted from local: {document_id}")
        
        if deleted_from_azure or deleted_from_local:
            return {
                "message": "Document deleted successfully",
                "document_id": document_id,
                "azure_deleted": deleted_from_azure,
                "local_deleted": deleted_from_local
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get document statistics with category breakdown"""
    try:
        categories = {
            "PDF": 0,
            "Word": 0,
            "Excel": 0,
            "CSV": 0,
            "PowerPoint": 0,
            "Text": 0,
            "General": 0
        }
        total_documents = 0
        
        # Get counts from Azure
        if search_client:
            try:
                # Get total count
                total_documents = search_client.get_document_count()
                
                # Get category facets
                for category in categories.keys():
                    filter_query = f"category eq '{category}'"
                    results = search_client.search(
                        search_text="*",
                        filter=filter_query,
                        include_total_count=True,
                        top=0  # We only need count
                    )
                    count = results.get_count() if hasattr(results, 'get_count') else 0
                    categories[category] = count
                    
            except Exception as e:
                logger.warning(f"Could not get Azure stats: {e}")
        
        # Fallback to local if Azure fails
        if total_documents == 0 and os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                docs = json.load(f)
            total_documents = len(docs)
            for doc in docs:
                cat = doc.get("category", "General")
                if cat in categories:
                    categories[cat] += 1
        
        return {
            "total_documents": total_documents,
            "categories": categories,
            "index_name": INDEX_NAME,
            "status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"total_documents": 0, "categories": {}, "error": str(e)}

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 IntelliDocs AI - Azure AI Search Portal")
    print("="*60)
    print(f"Mode: AZURE PRIMARY")
    print(f"Endpoint: {AZURE_SEARCH_ENDPOINT}")
    print(f"Index: {INDEX_NAME}")
    print(f"Fallback: {'Enabled' if FALLBACK_TO_LOCAL else 'Disabled'}")
    print("="*60)
    print(f"API: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
 


###########################################################################


"""
IntelliDocs AI - Azure AI Search PRIMARY Implementation
Institute Project Version (Patched for SemanticSettings + .env)
"""

import os
import json
import uuid
import logging
import sys
import logging 
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------
# Load .env from project root
# ---------------------------
from dotenv import load_dotenv

# Determine project root: backend/main.py -> parents[2] reaches project root
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print(f"⚠️ .env not found at {env_path} — ensure your .env is at project root")

# Configure logging for Azure
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# ---------------------------
# Document processing imports
# ---------------------------
try:
    import PyPDF2
    from docx import Document as DocxDocument
    from pptx import Presentation
    import openpyxl
    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False
    print("⚠️ Install document processors: pip install PyPDF2 python-docx openpyxl python-pptx")

# ---------------------------
# Azure SDK imports (compat)
# ---------------------------
try:
    # Core Azure Search imports
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    # Try to import models that may vary across versions
    from azure.search.documents.indexes.models import (
        SearchIndex,
        SimpleField,
        SearchableField,
        SearchFieldDataType,
        SearchField,
        VectorSearch,
        SemanticConfiguration,
        SemanticField,
        SearchSuggester,
    )

    # Try to import SemanticSettings — if missing, we'll create a compatibility shim below
    try:
        from azure.search.documents.indexes.models import SemanticSettings  # may not exist in newer SDKs
        SEMANTIC_SETTINGS_AVAILABLE = True
    except Exception:
        SemanticSettings = None
        SEMANTIC_SETTINGS_AVAILABLE = False

    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError

except Exception as e:
    # Provide a clear message with the recommended versions
    raise ImportError(
        "❌ Azure SDK import failed or version mismatch.\n"
        "Recommended fix (in your activated .venv):\n"
        "pip install 'azure-search-documents==11.4.0' 'azure-core==1.29.4' 'azure-common==1.1.28'\n"
        f"Original error: {e}"
    ) from e

# If SemanticSettings is not available in installed SDK, define a small compatibility class
if not SEMANTIC_SETTINGS_AVAILABLE:
    class SemanticSettings:
        """
        Compatibility shim for SemanticSettings if it's not present in the installed Azure SDK.
        This class only stores configurations and mirrors the expected interface minimally.
        """
        def __init__(self, configurations: Optional[List[Any]] = None):
            self.configurations = configurations or []

        def __repr__(self):
            return f"SemanticSettings(configurations={self.configurations!r})"

# ---------------------------
# Configure logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Load environment variables (after dotenv)
# ---------------------------
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_QUERY_KEY = os.getenv("AZURE_SEARCH_QUERY_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "employee-documents")
USE_AZURE_PRIMARY = os.getenv("USE_AZURE_PRIMARY", "true").lower() == "true"
FALLBACK_TO_LOCAL = os.getenv("FALLBACK_TO_LOCAL", "true").lower() == "true"

# Print environment values for debugging (non-sensitive)
print("🔍 Environment check:")
print("  AZURE_SEARCH_ENDPOINT =", AZURE_SEARCH_ENDPOINT)
print("  AZURE_SEARCH_INDEX_NAME =", INDEX_NAME)
print("  USE_AZURE_PRIMARY =", USE_AZURE_PRIMARY)
print("  FALLBACK_TO_LOCAL =", FALLBACK_TO_LOCAL)

# Validate Azure Configuration
if USE_AZURE_PRIMARY and (not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_ADMIN_KEY):
    raise ValueError(
        "❌ AZURE CREDENTIALS MISSING!\n"
        "Please update your .env file with:\n"
        "- AZURE_SEARCH_ENDPOINT\n"
        "- AZURE_SEARCH_ADMIN_KEY\n\n"
        "This project REQUIRES Azure AI Search as primary service.\n"
    )

# ---------------------------
# Local backup + upload dir
# ---------------------------
DOCUMENTS_DB_FILE = "documents_db_backup.json"
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(
    title="IntelliDocs AI - Azure AI Search Portal",
    description="Intelligent document search powered by Azure AI Search (PRIMARY)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Initialize Azure clients (PRIMARY)
# ---------------------------
index_client = None
search_client = None

if USE_AZURE_PRIMARY:
    try:
        admin_credential = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        query_credential = AzureKeyCredential(AZURE_SEARCH_QUERY_KEY or AZURE_SEARCH_ADMIN_KEY)

        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=admin_credential
        )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            credential=query_credential
        )

        logger.info("✅ Azure AI Search clients initialized - PRIMARY MODE")

    except Exception as e:
        logger.error(f"❌ Failed to initialize Azure AI Search: {e}")
        # If fallback is allowed, continue; otherwise raise
        if not FALLBACK_TO_LOCAL:
            raise

# ---------------------------
# Pydantic Models
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    top: Optional[int] = 10
    skip: Optional[int] = 0
    filter: Optional[str] = None
    orderby: Optional[str] = None
    search_mode: Optional[str] = "any"  # any or all
    query_type: Optional[str] = "simple"  # simple or full

class SearchResponse(BaseModel):
    count: int
    results: List[dict]
    facets: Optional[dict] = None
    suggestions: Optional[List[str]] = None

class IndexStats(BaseModel):
    document_count: int
    storage_size: int
    index_name: str
    status: str

# ---------------------------
# Index creation / update
# ---------------------------
async def create_or_update_index():
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String,
                        sortable=True, filterable=True, facetable=True),
        SearchableField(name="content", type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),
        SimpleField(name="category", type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
        SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset,
                    filterable=True, sortable=True),
        SimpleField(name="file_size", type=SearchFieldDataType.Int64,
                    filterable=True, sortable=True),
        SimpleField(name="word_count", type=SearchFieldDataType.Int64,
                    filterable=True, sortable=True),
        SearchableField(name="tags", collection=True, type=SearchFieldDataType.String,
                        filterable=True, facetable=True)
    ]

    suggester = SearchSuggester(name="sg", source_fields=["title", "content"])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        suggesters=[suggester]
    )

    try:
        result = index_client.create_index(index)
        logger.info(f"✅ Created new index: {INDEX_NAME}")
        return result
    except Exception as e:
        if "already exists" in str(e).lower():
            result = index_client.create_or_update_index(index)
            logger.info(f"✅ Updated existing index: {INDEX_NAME}")
            return result
        else:
            logger.error(f"❌ Error creating index: {e}")
            raise

# ---------------------------
# API endpoints
# ---------------------------
@app.on_event("startup")
async def startup_event():
    try:
        if USE_AZURE_PRIMARY and index_client is not None:
            await create_or_update_index()
            logger.info("✅ Azure Search index ready")
    except Exception as e:
        logger.error(f"❌ Failed to setup index: {e}")

@app.get("/")
async def root():
    return {
        "message": "IntelliDocs AI - Azure AI Search Portal",
        "status": "running",
        "mode": "AZURE PRIMARY" if USE_AZURE_PRIMARY else "LOCAL",
        "azure_endpoint": AZURE_SEARCH_ENDPOINT,
        "index_name": INDEX_NAME,
        "version": "3.0.0",
        "endpoints": {
            "search": "/api/search",
            "upload": "/api/documents/upload-file",
            "index_stats": "/api/index/stats",
            "index_create": "/api/index/create",
            "suggestions": "/api/search/suggestions",
            "docs": "/docs"
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_documents_azure_primary(request: SearchRequest):
    try:
        # Check if Azure search client is available
        if not search_client:
            if FALLBACK_TO_LOCAL:
                return await search_local_fallback(request)
            else:
                raise HTTPException(
                    status_code=503,
                    detail="Azure Search not available and local fallback is disabled"
                )
            
        logger.info(f"🔍 Azure Search Query: {request.query}")

        # Configure Azure search options
        search_options = {
            "search_text": request.query,
            "search_mode": request.search_mode,
            "include_total_count": True,
            "top": request.top,
            "skip": request.skip,
            "select": ["id", "title", "category", "created_date", "file_size", "word_count", "content", "tags"]
        }

        if request.filter:
            search_options["filter"] = request.filter

        if request.orderby:
            search_options["order_by"] = request.orderby.split(",")

        # Execute Azure search
        azure_results = search_client.search(**search_options)

        results = []
        for result in azure_results:
            doc = {
                "id": result.get("id"),
                "title": result.get("title"),
                "content": result.get("content", ""),  # Return FULL content
                "category": result.get("category"),
                "created_date": result.get("created_date"),
                "file_size": result.get("file_size"),
                "word_count": result.get("word_count"),
                "score": result.get("@search.score", 0),
                "highlights": result.get("@search.highlights", {})
            }
            results.append(doc)

        count = azure_results.get_count() if hasattr(azure_results, 'get_count') else len(results)

        logger.info(f"✅ Azure Search returned {count} results")

        return SearchResponse(
            count=count,
            results=results
        )

    except Exception as azure_error:
        logger.error(f"❌ Azure Search failed: {azure_error}")
        if FALLBACK_TO_LOCAL:
            logger.info("⚠️ Falling back to local search")
            return await search_local_fallback(request)
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Azure Search unavailable: {str(azure_error)}"
            )

async def search_local_fallback(request: SearchRequest):
    """
    Local search fallback - only used when Azure is unavailable
    """
    try:
        # Load local documents
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, 'r', encoding='utf-8') as f:
                documents = json.load(f)
        else:
            documents = []

        # Convert search query to lowercase for case-insensitive search
        query_lower = request.query.lower()
        results = []

        # Search through local documents
        for doc in documents:
            title_match = query_lower in doc.get("title", "").lower()
            content_match = query_lower in doc.get("content", "").lower()
            
            if title_match or content_match:
                # Calculate a simple score
                score = 2.0 if title_match else 1.0
                
                results.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "content": doc.get("content", ""),  # Return FULL content
                    "category": doc.get("category"),
                    "created_date": doc.get("created_date"),
                    "file_size": doc.get("file_size", 0),
                    "word_count": doc.get("word_count", 0),
                    "score": score,
                    "highlights": {}
                })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Apply pagination
        start = request.skip if request.skip else 0
        end = start + request.top if request.top else len(results)
        paginated_results = results[start:end]

        logger.info(f"📁 Local search returned {len(paginated_results)} results")

        return SearchResponse(
            count=len(results),
            results=paginated_results
        )
        
    except Exception as e:
        logger.error(f"Local search failed: {e}")
        return SearchResponse(count=0, results=[])

@app.post("/api/documents/upload-file")
async def upload_file_azure_primary(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()

        with open(file_path, "wb") as buffer:
            buffer.write(content)

        # Process document to extract text
        doc_data = DocumentProcessor.process_document(str(file_path))
        
        # Generate unique ID and timestamp
        doc_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc).isoformat()

        # Get full content
        full_content = doc_data["content"]
        
        # Create document object
        azure_document = {
            "id": doc_id,
            "title": doc_data["filename"],
            "content": full_content,  # Full content
            "category": doc_data["doc_type"],
            "created_date": current_time,
            "file_size": doc_data.get("file_size", 0),
            "word_count": doc_data.get("word_count", 0),
            "tags": [doc_data["doc_type"], "uploaded"]
        }

        # Try to upload to Azure (PRIMARY)
        upload_success = False
        if search_client:
            try:
                # Azure free tier has 32KB limit per field
                azure_doc_to_upload = azure_document.copy()
                
                # Check if content needs truncation for Azure
                if len(azure_doc_to_upload["content"]) > 30000:
                    azure_doc_to_upload["content"] = azure_doc_to_upload["content"][:30000]
                    logger.info(f"Content truncated for Azure: {len(full_content)} -> 30000 chars")
                
                result = search_client.upload_documents(documents=[azure_doc_to_upload])
                
                if result[0].succeeded:
                    logger.info(f"✅ Document uploaded to Azure: {doc_id}")
                    upload_success = True
                else:
                    logger.error(f"Azure upload failed: {result[0].error}")
                    
            except Exception as e:
                logger.error(f"Azure upload exception: {e}")

        # Backup to local storage (with FULL content)
        if FALLBACK_TO_LOCAL or not upload_success:
            save_to_local_backup(azure_document)  # This saves FULL content locally

        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file: {e}")

        return {
            "message": "Document indexed successfully",
            "document_id": doc_id,
            "filename": doc_data["filename"],
            "category": doc_data["doc_type"],
            "azure_indexed": upload_success,
            "word_count": doc_data.get("word_count", 0),
            "file_size": doc_data.get("file_size", 0),
            "content_length": len(full_content)
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/index/stats", response_model=IndexStats)
async def get_index_statistics():
    try:
        index = index_client.get_index(INDEX_NAME)
        count_result = search_client.get_document_count()

        return IndexStats(
            document_count=count_result,
            storage_size=0,
            index_name=INDEX_NAME,
            status="healthy"
        )
    except Exception as e:
        logger.error(f"Failed to get index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index/create")
async def create_index_endpoint():
    try:
        result = await create_or_update_index()
        return {
            "message": "Index created/updated successfully",
            "index_name": INDEX_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/suggestions")
async def get_search_suggestions(query: str, top: int = 5):
    try:
        suggestions = search_client.suggest(
            search_text=query,
            suggester_name="sg",
            top=top
        )
        return {
            "suggestions": [s["title"] for s in suggestions]
        }
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        return {"suggestions": []}

def save_to_local_backup(document: dict):
    try:
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, 'r', encoding='utf-8') as f:
                docs = json.load(f)
        else:
            docs = []

        docs.append(document)

        with open(DOCUMENTS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2)

        logger.info("📁 Document backed up locally")
    except Exception as e:
        logger.error(f"Local backup failed: {e}")

@app.get("/api/documents/recent")
async def get_recent_documents(limit: int = 10):
    """Get recently uploaded documents with FULL content"""
    try:
        documents = []
        
        # Try Azure first (PRIMARY)
        if search_client:
            try:
                results = search_client.search(
                    search_text="*",
                    order_by=["created_date desc"],
                    top=limit,
                    include_total_count=True,
                    select=["id", "title", "category", "created_date", "file_size", "word_count", "content", "tags"]
                )
                
                for doc in results:
                    documents.append({
                        "id": doc.get("id"),
                        "title": doc.get("title"),
                        "category": doc.get("category"),
                        "created_date": doc.get("created_date"),
                        "file_size": doc.get("file_size"),
                        "word_count": doc.get("word_count"),
                        "content": doc.get("content", ""),  # Return FULL content
                        "tags": doc.get("tags", [])
                    })
                    
            except Exception as e:
                logger.warning(f"Could not fetch from Azure: {e}")
        
        # Fallback to local if no Azure results
        if not documents and os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                local_docs = json.load(f)
            
            local_docs = sorted(
                local_docs, 
                key=lambda x: x.get("created_date", ""), 
                reverse=True
            )[:limit]
            
            for doc in local_docs:
                documents.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "category": doc.get("category"),
                    "created_date": doc.get("created_date"),
                    "file_size": doc.get("file_size"),
                    "word_count": doc.get("word_count"),
                    "content": doc.get("content", ""),  # Return FULL content
                    "tags": doc.get("tags", [])
                })
        
        return {"documents": documents, "count": len(documents)}
        
    except Exception as e:
        logger.error(f"Error fetching recent documents: {e}")
        return {"documents": [], "error": str(e)}

@app.get("/api/documents/{document_id}")
async def get_document_by_id(document_id: str):
    """Get a single document with full content by ID"""
    try:
        logger.info(f"📄 Fetching document: {document_id}")
        document = None
        
        # Try to get from Azure first (PRIMARY)
        if search_client:
            try:
                result = search_client.get_document(key=document_id)
                if result:
                    document = {
                        "id": result.get("id"),
                        "title": result.get("title"),
                        "content": result.get("content", ""),  # Full content
                        "category": result.get("category"),
                        "created_date": result.get("created_date"),
                        "file_size": result.get("file_size", 0),
                        "word_count": result.get("word_count", 0),
                        "tags": result.get("tags", [])
                    }
                    logger.info(f"✅ Found document in Azure: {document_id}")
            except ResourceNotFoundError:
                logger.warning(f"Document not found in Azure: {document_id}")
            except Exception as e:
                logger.warning(f"Azure fetch error: {e}")
        
        # Fallback to local backup if not found in Azure
        if not document and os.path.exists(DOCUMENTS_DB_FILE):
            logger.info(f"🔍 Searching local backup for: {document_id}")
            try:
                with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                    local_docs = json.load(f)
                
                for doc in local_docs:
                    if doc.get("id") == document_id:
                        document = {
                            "id": doc.get("id"),
                            "title": doc.get("title"),
                            "content": doc.get("content", ""),  # Full content
                            "category": doc.get("category"),
                            "created_date": doc.get("created_date"),
                            "file_size": doc.get("file_size", 0),
                            "word_count": doc.get("word_count", 0),
                            "tags": doc.get("tags", [])
                        }
                        logger.info(f"✅ Found document in local backup: {document_id}")
                        break
            except Exception as e:
                logger.error(f"Error reading local backup: {e}")
        
        if document:
            return document
        else:
            logger.error(f"❌ Document not found: {document_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Document not found: {document_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching document {document_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching document: {str(e)}"
        )


@app.delete("/api/documents/{document_id}")
async def delete_document_by_id(document_id: str):
    """Delete a document from Azure Search and local backup"""
    try:
        logger.info(f"🗑️ Deleting document: {document_id}")
        deleted_from_azure = False
        deleted_from_local = False
        
        # Delete from Azure
        if search_client:
            try:
                search_client.delete_documents(documents=[{"id": document_id}])
                logger.info(f"✅ Deleted from Azure: {document_id}")
                deleted_from_azure = True
            except Exception as e:
                logger.warning(f"Azure delete issue: {e}")
        
        # Delete from local backup
        if os.path.exists(DOCUMENTS_DB_FILE):
            try:
                with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                
                original_count = len(docs)
                docs = [d for d in docs if d.get("id") != document_id]
                
                if len(docs) < original_count:
                    with open(DOCUMENTS_DB_FILE, "w", encoding="utf-8") as f:
                        json.dump(docs, f, indent=2)
                    deleted_from_local = True
                    logger.info(f"✅ Deleted from local: {document_id}")
            except Exception as e:
                logger.error(f"Local delete error: {e}")
        
        if deleted_from_azure or deleted_from_local:
            return {
                "message": "Document deleted successfully",
                "document_id": document_id,
                "azure_deleted": deleted_from_azure,
                "local_deleted": deleted_from_local
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail="Document not found in Azure or local storage"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting document: {str(e)}"
        )

@app.delete("/api/documents/{document_id}")  # Fixed path
async def delete_document(document_id: str):
    """Delete a document from Azure Search and local backup"""
    try:
        deleted_from_azure = False
        deleted_from_local = False
        
        # Delete from Azure
        if search_client:
            try:
                search_client.delete_documents(documents=[{"id": document_id}])
                logger.info(f"✅ Deleted from Azure: {document_id}")
                deleted_from_azure = True
            except Exception as e:
                logger.warning(f"Azure delete issue: {e}")
        
        # Delete from local backup
        if os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                docs = json.load(f)
            
            original_count = len(docs)
            docs = [d for d in docs if d.get("id") != document_id]
            
            if len(docs) < original_count:
                with open(DOCUMENTS_DB_FILE, "w", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2)
                deleted_from_local = True
                logger.info(f"✅ Deleted from local: {document_id}")
        
        if deleted_from_azure or deleted_from_local:
            return {
                "message": "Document deleted successfully",
                "document_id": document_id,
                "azure_deleted": deleted_from_azure,
                "local_deleted": deleted_from_local
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get document statistics with category breakdown"""
    try:
        categories = {
            "PDF": 0,
            "Word": 0,
            "Excel": 0,
            "CSV": 0,
            "PowerPoint": 0,
            "Text": 0,
            "General": 0
        }
        total_documents = 0
        
        # Get counts from Azure
        if search_client:
            try:
                # Get total count
                total_documents = search_client.get_document_count()
                
                # Get category facets
                for category in categories.keys():
                    filter_query = f"category eq '{category}'"
                    results = search_client.search(
                        search_text="*",
                        filter=filter_query,
                        include_total_count=True,
                        top=0  # We only need count
                    )
                    count = results.get_count() if hasattr(results, 'get_count') else 0
                    categories[category] = count
                    
            except Exception as e:
                logger.warning(f"Could not get Azure stats: {e}")
        
        # Fallback to local if Azure fails
        if total_documents == 0 and os.path.exists(DOCUMENTS_DB_FILE):
            with open(DOCUMENTS_DB_FILE, "r", encoding="utf-8") as f:
                docs = json.load(f)
            total_documents = len(docs)
            for doc in docs:
                cat = doc.get("category", "General")
                if cat in categories:
                    categories[cat] += 1
        
        return {
            "total_documents": total_documents,
            "categories": categories,
            "index_name": INDEX_NAME,
            "status": "connected"
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"total_documents": 0, "categories": {}, "error": str(e)}

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 IntelliDocs AI - Azure AI Search Portal")
    print("="*60)
    print(f"Mode: AZURE PRIMARY")
    print(f"Endpoint: {AZURE_SEARCH_ENDPOINT}")
    print(f"Index: {INDEX_NAME}")
    print(f"Fallback: {'Enabled' if FALLBACK_TO_LOCAL else 'Disabled'}")
    print("="*60)
    print(f"API: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
