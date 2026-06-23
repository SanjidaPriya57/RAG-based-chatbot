# AI Finance Assistant Chatbot

A lightweight finance assistant proof-of-concept using a FastAPI backend, RAG-style document retrieval, PostgreSQL, and Redis.

## Project structure

- `docker-compose.yaml` - local stack for the app, PostgreSQL, and Redis.
- `rag-demo/` - Python service and application source code.
  - `Dockerfile` - container build image for the FastAPI app with hot reload support.
  - `requirements.txt` - Python dependencies.
  - `main.py` - FastAPI entrypoint with CORS middleware and route registration.
  - `db.py` - PostgreSQL connection pool and table initialization.
  - `memory/cache.py` - Redis caching layer for query results and document metadata.
  - `rag/chunker.py` - Document chunking utilities for text and PDF ingestion.
  - `rag/embeddings.py` - Embedding service for text vectorization.
  - `rag/retriever.py` - Semantic search and relevance ranking over document chunks.
  - `routes/upload.py` - Document upload and ingestion endpoint.
  - `routes/ask.py` - Question-answering endpoint with RAG pipeline.

## Dependencies

The app depends on:

- `fastapi` - web framework
- `uvicorn` - ASGI server
- `anthropic` - Claude API integration
- `asyncpg` - PostgreSQL async driver
- `redis` - cache backend
- `pypdf2` - PDF parsing
- `python-multipart` - form file uploads
- `numpy` - numerical operations

## Local development

### Prepare environment

1. Navigate to the project root.
2. Create and activate a Python virtual environment:
   - **Windows PowerShell**: 
     ```powershell
     python -m venv rag-demo\.venv
     .\rag-demo\.venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**: 
     ```bash
     python3 -m venv rag-demo/.venv
     source rag-demo/.venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r rag-demo/requirements.txt
   ```

4. Set up environment variables (copy and configure):
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your `ANTHROPIC_API_KEY`.

### Run locally with hot reload

From `rag-demo/`:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at `http://127.0.0.1:8000` and interactive docs at `http://127.0.0.1:8000/docs`.

### Run with Docker Compose (with hot reload)

From the repository root:

```bash
docker compose up --build
```

This starts:

- **App service** on `http://localhost:8000` (with auto-reload on file changes)
- **PostgreSQL** on `localhost:5432`
- **Redis** on `localhost:6379`

**Hot Reload**: The Dockerfile now includes volume mounts and runs `uvicorn` with `--reload`. Any changes to files in `rag-demo/` will automatically restart the service.

## API Endpoints

### Health & Info

- **GET `/`** - Service info and API documentation links
- **GET `/health`** - Health check endpoint

### Document Management

- **POST `/api/upload`** - Upload a PDF document for ingestion
  - Request: Multipart form with `file` field (PDF only)
  - Response: Document ID, filename, and chunk count

- **GET `/api/documents`** - List all uploaded documents
  - Response: Array of document metadata

### Question-Answering

- **POST `/api/ask`** - Ask a question over indexed documents
  - Request: `{"query": "What is the Q3 revenue?", "top_k": 5}`
  - Response: `{"query": "...", "answer": "...", "sources": [...]}`

## Example Usage

### Upload a document

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@finance_report.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key financial metrics?"}'
```

## Environment configuration

Create a `.env` file based on `.env.example`:

```env
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/finance_db
POSTGRES_HOST=db
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=finance_db

# Redis
REDIS_URL=redis://redis:6379/0

# Anthropic API (required for Q&A)
ANTHROPIC_API_KEY=sk-ant-...

# Logging
LOG_LEVEL=INFO
```

## Developer Workflow

1. **Start the stack**: `docker compose up --build`
2. **Edit files**: Changes to any Python files in `rag-demo/` trigger automatic reload
3. **Check logs**: `docker compose logs -f ai-finance-app`
4. **Upload documents**: Use `/api/upload` to ingest PDFs
5. **Test Q&A**: Use `/api/ask` to query documents
6. **Inspect database**: Connect to PostgreSQL on `localhost:5432`
7. **Monitor cache**: Redis CLI at `localhost:6379`

## Key Features

- **RAG Pipeline**: Chunks documents, generates embeddings, and retrieves relevant context
- **Semantic Search**: Uses cosine similarity to find relevant document sections
- **LLM Integration**: Claude generates contextual answers based on retrieved documents
- **Caching**: Redis caches answers and document metadata for performance
- **Hot Reload**: Docker-based development with automatic file change detection
- **Async Support**: Fully async FastAPI with connection pooling

## Database Schema

### documents
- `id` (primary key)
- `filename` - uploaded filename
- `content` - full document text
- `uploaded_at`, `created_at` - timestamps

### document_chunks
- `id` (primary key)
- `document_id` (foreign key)
- `chunk_text` - text snippet
- `chunk_index` - position in document
- `embedding` - vector representation

### conversations
- `id` (primary key)
- `user_id` - optional user identifier
- `message` - user query
- `response` - assistant answer
- `created_at` - timestamp

## Notes

- The Dockerfile now starts `uvicorn` automatically with `--reload` flag
- Volume mounts ensure hot reload works in Docker
- ANTHROPIC_API_KEY is required for Q&A functionality
- PostgreSQL and Redis start automatically with `docker compose up`
- The app waits for database initialization before serving requests

## Next steps

- Add authentication and user session management
- Implement streaming responses for long-running queries
- Add document summarization endpoints
- Improve embedding quality with production embeddings service
- Add rate limiting and usage analytics
- Set up monitoring and error tracking
- Add comprehensive test suite

