"""
E-Cell AI & Automation Task 2 — Full RAG Pipeline
===================================================
Built following the pixegami LangChain RAG tutorial (youtube.com/watch?v=tcqEUSNCn8I)
Extended with all 5 stages required by the task.

Stack:
  • LangChain       — document loading, splitting, prompt templating
  • Google Gemini   — embeddings (embedding-001) + chat (gemini-1.5-flash) [FREE]
  • ChromaDB        — persistent local vector store
  • FastAPI         — Stage 5 API deployment

Folder layout:
    Ecell/
    ├── pipeline.py
    ├── .env          ← GOOGLE_API_KEY=your_key_here
    ├── data/         ← your PDFs here
    └── models/       ← auto-created

Install:
    pip install langchain langchain-community langchain-google-genai
                langchain-text-splitters chromadb pypdf google-generativeai
                python-dotenv fastapi uvicorn numpy

Get free Gemini API key: https://aistudio.google.com/app/apikey

Run:
    python pipeline.py --run-all
    python pipeline.py --stage 1
    python pipeline.py --stage 2
    python pipeline.py --stage 3 --query "What is X?"
    python pipeline.py --stage 3 --interactive
    python pipeline.py --stage 4
    python pipeline.py --stage 5
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import logging
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()   # reads GOOGLE_API_KEY from .env file

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
CHROMA_PATH = MODELS_DIR / "chroma"
EVAL_PATH   = MODELS_DIR / "eval_results.json"

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 500
TOP_K            = 3
SIMILARITY_FLOOR = 0.3    # Gemini embeddings score differently — lower floor

EMBED_MODEL = "gemini-embedding-001"     # free Gemini embedding model
CHAT_MODEL  = "gemini-1.5-flash"         # free Gemini chat model

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

STRICT_PROMPT_TEMPLATE = """
Answer the question based ONLY on the following context.
Do NOT use any outside knowledge.
If the answer is not in the context, say exactly:
"I don't have enough information in the provided documents to answer this."
Always cite the source document and page number inline.

Context:
{context}

---

Question: {question}

Answer (cite sources as [Source: filename, Page: N]):
"""

DOC_TYPE_PATTERNS = {
    "Standard Operational Procedure": [
        r"\bSOP\b", r"standard operating procedure",
        r"step-by-step", r"work instruction",
    ],
    "Corporate Policy Framework": [
        r"\bpolicy\b", r"corporate policy", r"governance", r"framework",
    ],
    "Compliance Regulation": [
        r"\bcompliance\b", r"\bregulation\b", r"\bGDPR\b",
        r"\bISO\b", r"\baudit\b", r"regulatory",
    ],
    "Technical Troubleshooting Log": [
        r"\berror\b", r"\bbug\b", r"troubleshoot",
        r"incident", r"root cause", r"resolution",
    ],
}

# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class RAGResponse:
    query      : str
    answer     : str
    confidence : float
    sources    : List[Dict]
    latency_s  : float

# =============================================================================
# HELPERS
# =============================================================================

def _check_gemini_key():
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set.\n"
            "1. Get a free key at: https://aistudio.google.com/app/apikey\n"
            "2. Add it to your .env file:  GOOGLE_API_KEY=your_key_here\n"
            "   Or in PowerShell:  $env:GOOGLE_API_KEY='your_key_here'"
        )
    return key


def clean_text(text: str) -> str:
    """Remove PDF layout noise."""
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"(?m)^\s*(Page\s+\d+\s+of\s+\d+|\d+)\s*$", "", text)
    for pat in [
        r"(?i)confidential[^\n]*",
        r"(?i)all rights reserved[^\n]*",
        r"(?i)internal use only[^\n]*",
    ]:
        text = re.sub(pat, "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def classify_doc_type(text: str) -> str:
    tl = text.lower()
    for doc_type, patterns in DOC_TYPE_PATTERNS.items():
        if any(re.search(p, tl) for p in patterns):
            return doc_type
    return "Technical Document"


def get_embedding_function():
    """Google Gemini embeddings — free tier."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    key = _check_gemini_key()
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=key,
    )


def get_chat_model():
    """Google Gemini 1.5 Flash — free tier."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = _check_gemini_key()
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=key,
        temperature=0.1,
        convert_system_message_to_human=True,
    )


# =============================================================================
# STAGE 1 — DOCUMENT INGESTION & TEXT SEGMENTATION
# =============================================================================

def load_documents(data_path: Path = DATA_PATH) -> list:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

    log.info(f"Loading PDFs from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_path}\n"
            f"Create it and add your PDFs inside."
        )

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=False,
        use_multithreading=False,
    )
    documents = loader.load()

    if not documents:
        raise FileNotFoundError(f"No PDFs found in: {data_path}")

    log.info(f"Loaded {len(documents)} raw pages")

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    before    = len(documents)
    documents = [d for d in documents if d.page_content.strip()]
    if before - len(documents):
        log.info(f"Dropped {before - len(documents)} empty pages after cleaning")

    return documents


def split_text(documents: list) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    log.info(f"Split {len(documents)} pages into {len(chunks)} chunks")

    if chunks:
        sample = chunks[10] if len(chunks) > 10 else chunks[0]
        log.info(f"Sample chunk: {sample.page_content[:200]!r}")
        log.info(f"Sample metadata: {sample.metadata}")

    return chunks


def run_stage1(data_path: Path = DATA_PATH) -> list:
    log.info("=" * 55)
    log.info("STAGE 1: Document Ingestion & Text Segmentation")
    log.info("=" * 55)

    documents = load_documents(data_path)
    chunks    = split_text(documents)

    for chunk in chunks:
        chunk.metadata["doc_type"] = classify_doc_type(chunk.page_content)

    log.info(f"Stage 1 complete: {len(chunks)} chunks ready")
    return chunks


# =============================================================================
# STAGE 2 — EMBEDDING GENERATION & INDEXING
# =============================================================================

def save_to_chroma(chunks: list) -> None:
    from langchain_community.vectorstores import Chroma

    _check_gemini_key()

    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
        log.info("Cleared existing Chroma DB")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    log.info(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")

    # Gemini free tier has rate limits — embed in small batches with a short pause
    BATCH = 10
    all_chunks = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
        log.info(f"  Batch {i // BATCH + 1}/{-(-len(chunks) // BATCH)} ({len(batch)} chunks)")
        if i == 0:
            db = Chroma.from_documents(
                batch,
                get_embedding_function(),
                persist_directory=str(CHROMA_PATH),
            )
        else:
            db.add_documents(batch)
        time.sleep(15)   # respect Gemini free-tier rate limit

    db.persist()
    log.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")


def run_stage2(chunks: Optional[list] = None) -> None:
    log.info("=" * 55)
    log.info("STAGE 2: Embedding Generation & Indexing")
    log.info("=" * 55)

    if chunks is None:
        chunks = run_stage1()

    t0 = time.time()
    save_to_chroma(chunks)
    log.info(f"Stage 2 complete: {time.time()-t0:.1f}s  model={EMBED_MODEL}")


# =============================================================================
# STAGE 3 — LLM INFERENCE & CONTEXT ORCHESTRATION
# =============================================================================

def query_rag(query_text: str, strict: bool = True) -> RAGResponse:
    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate

    _check_gemini_key()

    if not CHROMA_PATH.exists():
        raise FileNotFoundError(
            "Chroma DB not found. Run stages 1 and 2 first:\n"
            "  python pipeline.py --stage 1\n"
            "  python pipeline.py --stage 2"
        )

    t0 = time.time()

    # Load DB
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
    )

    # Search — same as video: similarity_search_with_relevance_scores, k=3
    results = db.similarity_search_with_relevance_scores(query_text, k=TOP_K)

    # Similarity floor check
    if not results or results[0][1] < SIMILARITY_FLOOR:
        log.warning(f"No results above similarity floor ({SIMILARITY_FLOOR})")
        return RAGResponse(
            query      = query_text,
            answer     = "I don't have enough information in the provided documents to answer this question.",
            confidence = results[0][1] if results else 0.0,
            sources    = [],
            latency_s  = round(time.time() - t0, 3),
        )

    # Build context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Format prompt
    template    = STRICT_PROMPT_TEMPLATE if strict else PROMPT_TEMPLATE
    prompt_tmpl = ChatPromptTemplate.from_template(template)
    prompt      = prompt_tmpl.format(context=context_text, question=query_text)

    # Call Gemini
    model         = get_chat_model()
    response_text = model.predict(prompt)

    # Build sources
    sources = [
        {
            "source"  : Path(doc.metadata.get("source", "unknown")).name,
            "page"    : doc.metadata.get("page", 0),
            "doc_type": doc.metadata.get("doc_type", "Technical Document"),
            "score"   : round(score, 4),
        }
        for doc, score in results
    ]

    # Print like the video
    print(f"Response: {response_text}")
    print(f"Sources: {[s['source'] for s in sources]}")

    return RAGResponse(
        query      = query_text,
        answer     = response_text,
        confidence = round(results[0][1], 4),
        sources    = sources,
        latency_s  = round(time.time() - t0, 3),
    )


def print_response(resp: RAGResponse) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  QUERY      : {resp.query}")
    print(f"  CONFIDENCE : {resp.confidence:.2%}")
    print(f"  LATENCY    : {resp.latency_s}s")
    print(sep)
    print(f"\nResponse: {resp.answer}\n")
    if resp.sources:
        print("Sources:")
        for s in resp.sources:
            print(f"  - {s['source']}  page {s['page']}  [{s['doc_type']}]  score={s['score']:.4f}")
    print(f"{sep}\n")


def run_stage3(query: Optional[str] = None, interactive: bool = False) -> None:
    log.info("=" * 55)
    log.info("STAGE 3: LLM Inference & Context Orchestration")
    log.info("=" * 55)

    if interactive:
        print(f"\nRAG System ready  (model={CHAT_MODEL})")
        print("Type your question and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                q = input("Query: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                break
            try:
                print_response(query_rag(q))
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        q = query or "What is the main topic of the documents?"
        print_response(query_rag(q))


# =============================================================================
# STAGE 4 — PIPELINE EVALUATION
# =============================================================================

EVAL_QUERIES = [
    {
        "query"          : "What is the attention mechanism in transformers?",
        "expected_topics": ["attention", "query", "key", "value", "softmax"],
    },
    {
        "query"          : "How does BERT use masked language modelling?",
        "expected_topics": ["mask", "token", "pretrain", "bert"],
    },
    {
        "query"          : "What is retrieval augmented generation?",
        "expected_topics": ["retrieval", "generation", "document", "knowledge"],
    },
    {
        "query"          : "What are the contributions of the GPT-4 technical report?",
        "expected_topics": ["gpt", "capabilities", "evaluation", "multimodal"],
    },
    {
        "query"          : "What security controls does NIST recommend?",
        "expected_topics": ["access", "control", "authentication", "security"],
    },
]


def score_cr(docs: list, expected: List[str]) -> float:
    combined = " ".join(d.page_content.lower() for d in docs)
    hits = sum(1 for kw in expected if kw in combined)
    return round(hits / len(expected), 4) if expected else 0.0


def score_f(answer: str, docs: list) -> float:
    ctx_words = set(w.lower() for d in docs for w in d.page_content.split() if len(w) > 4)
    sents = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if not sents:
        return 0.0
    return round(sum(1 for s in sents if any(w.lower() in ctx_words for w in s.split())) / len(sents), 4)


def score_ar(answer: str, query: str) -> float:
    q_words = set(w.lower() for w in query.split() if len(w) > 3)
    if not q_words:
        return 0.0
    return round(sum(1 for w in q_words if w in answer.lower()) / len(q_words), 4)


def run_stage4() -> Dict:
    log.info("=" * 55)
    log.info("STAGE 4: Pipeline Evaluation")
    log.info("=" * 55)

    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate

    _check_gemini_key()
    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_function(),
    )

    results  = []
    resolved = 0

    for item in EVAL_QUERIES:
        q        = item["query"]
        expected = item["expected_topics"]
        log.info(f"Evaluating: {q[:60]}")

        t0   = time.time()
        hits = db.similarity_search_with_relevance_scores(q, k=TOP_K)
        lat  = round(time.time() - t0, 3)

        if not hits or hits[0][1] < SIMILARITY_FLOOR:
            docs   = []
            conf   = hits[0][1] if hits else 0.0
            answer = "No matching results."
        else:
            docs   = [doc for doc, _ in hits]
            conf   = round(hits[0][1], 4)
            ctx    = "\n\n---\n\n".join(d.page_content for d in docs)
            prompt = ChatPromptTemplate.from_template(STRICT_PROMPT_TEMPLATE).format(
                context=ctx, question=q
            )
            try:
                answer = get_chat_model().predict(prompt)
                resolved += 1
            except Exception as e:
                log.warning(f"LLM error: {e}")
                answer = ""

        time.sleep(15)   # Gemini free-tier rate limit between eval queries

        results.append({
            "query"     : q,
            "CR"        : score_cr(docs, expected),
            "F"         : score_f(answer, docs),
            "AR"        : score_ar(answer, q),
            "L"         : lat,
            "confidence": conf,
            "resolved"  : conf >= SIMILARITY_FLOOR,
        })

    qr  = round(resolved / len(EVAL_QUERIES), 4)
    avg = {
        "CR_avg": round(sum(r["CR"] for r in results) / len(results), 4),
        "F_avg" : round(sum(r["F"]  for r in results) / len(results), 4),
        "AR_avg": round(sum(r["AR"] for r in results) / len(results), 4),
        "L_avg" : round(sum(r["L"]  for r in results) / len(results), 3),
        "QR"    : qr,
    }

    w = 78
    print(f"\n{'='*w}")
    print(f"  STAGE 4 EVALUATION REPORT")
    print(f"  Chat: {CHAT_MODEL}  |  Embeddings: {EMBED_MODEL}  |  k={TOP_K}")
    print(f"{'='*w}")
    print(f"  {'Query':<43} {'CR':>6} {'F':>6} {'AR':>6} {'L(s)':>6} {'OK':>4}")
    print(f"  {'-'*43} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*4}")
    for r in results:
        tick = "YES" if r["resolved"] else "NO"
        print(f"  {r['query'][:42]:<43} {r['CR']:>6.3f} {r['F']:>6.3f} {r['AR']:>6.3f} {r['L']:>6.2f} {tick:>4}")
    print(f"  {'-'*43} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'AVERAGES':<43} {avg['CR_avg']:>6.3f} {avg['F_avg']:>6.3f} {avg['AR_avg']:>6.3f} {avg['L_avg']:>6.2f}")
    print(f"\n  QR (Query Resolution Rate): {qr:.0%}  ({resolved}/{len(EVAL_QUERIES)} resolved)")
    print(f"{'='*w}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"summary": avg, "per_query": results,
              "chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL}
    with open(EVAL_PATH, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Eval results saved: {EVAL_PATH}")
    return output


# =============================================================================
# STAGE 5 — FASTAPI DEPLOYMENT
# =============================================================================

def build_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(
        title="E-Cell RAG API",
        description="Semantic retrieval over technical PDFs — LangChain + Gemini.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    class QueryRequest(BaseModel):
        query: str

    class SourceItem(BaseModel):
        source  : str
        page    : int
        doc_type: str
        score   : float

    class QueryResponse(BaseModel):
        answer     : str
        confidence : float
        sources    : List[SourceItem]
        latency_s  : float

    @app.get("/", tags=["Health"])
    def root():
        return {"status": "ok", "docs": "/docs"}

    @app.get("/health", tags=["Health"])
    def health():
        return {
            "chroma_ready": CHROMA_PATH.exists(),
            "chat_model"  : CHAT_MODEL,
            "embed_model" : EMBED_MODEL,
        }

    @app.post("/query", response_model=QueryResponse, tags=["RAG"])
    def query_endpoint(req: QueryRequest):
        """
        Input : { "query": "What is X?" }
        Output: { "answer": "...", "confidence": 0.xx, "sources": [...] }
        """
        if not req.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        if not CHROMA_PATH.exists():
            raise HTTPException(status_code=503, detail="Run stages 1 and 2 first.")
        try:
            resp = query_rag(req.query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return QueryResponse(
            answer=resp.answer, confidence=resp.confidence,
            sources=[SourceItem(**s) for s in resp.sources],
            latency_s=resp.latency_s,
        )

    @app.get("/eval", tags=["Evaluation"])
    def get_eval():
        if not EVAL_PATH.exists():
            raise HTTPException(status_code=404, detail="Run stage 4 first.")
        with open(EVAL_PATH) as f:
            return json.load(f)

    return app


def run_stage5(host: str = "0.0.0.0", port: int = 8000) -> None:
    log.info("=" * 55)
    log.info("STAGE 5: API Deployment")
    log.info("=" * 55)

    try:
        import uvicorn
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    _check_gemini_key()
    app = build_app()

    print(f"\nRAG API starting ...")
    print(f"  Local :   http://localhost:{port}")
    print(f"  Docs  :   http://localhost:{port}/docs")
    print(f"  Health:   http://localhost:{port}/health")
    print(f"\n  Test it:")
    print(f'  Invoke-RestMethod -Uri http://localhost:{port}/query -Method POST -ContentType "application/json" -Body \'{{"query":"What is attention?"}}\'\n')

    uvicorn.run(app, host=host, port=port, log_level="info")


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_all(data_path: Path = DATA_PATH) -> None:
    chunks = run_stage1(data_path=data_path)
    run_stage2(chunks=chunks)
    run_stage3(query="What is the main topic of the documents?")
    run_stage4()
    run_stage5()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E-Cell RAG Pipeline — LangChain + Gemini (all 5 stages)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a single stage (1-5)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 5 stages end-to-end")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to PDF folder (default: ./data)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Query for Stage 3")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Stage 3: interactive REPL")
    parser.add_argument("--port", type=int, default=8000,
                        help="API port (default: 8000)")

    args      = parser.parse_args()
    data_path = Path(args.data_path) if args.data_path else DATA_PATH

    if args.run_all:
        run_all(data_path=data_path)
    elif args.stage == 1:
        run_stage1(data_path=data_path)
    elif args.stage == 2:
        run_stage2()
    elif args.stage == 3:
        run_stage3(query=args.query, interactive=args.interactive)
    elif args.stage == 4:
        run_stage4()
    elif args.stage == 5:
        run_stage5(port=args.port)
    else:
        parser.print_help()
