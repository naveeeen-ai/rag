import os
from typing import List, Tuple, Optional
from collections import Counter
import re
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv
load_dotenv()


CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")
MODEL = os.getenv("MODEL", "gpt-4o-mini")

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question using only the provided context.
If the answer is not contained in the context, say "I'm unaware of that topic because your context doesn't provide any information about it."

Context:
{context}

Question: {question}
"""

SUMMARY_PROMPT = """
You are a helpful tutor. Summarize the requested topic for students using ONLY the provided context.
- Be concise and accurate
- Structure with a short title, sections, and bullet points
- Include key definitions, formulas, and examples if present
- Keep it under 200-250 words

Topic: {topic}
Context:
{context}
"""

MINDMAP_PROMPT = """
You are a helpful tutor. Create a Mermaid mindmap covering the topic using ONLY the provided context.
- Use the exact Mermaid syntax starting with 'mindmap' on the first line
- Use the topic as the root node
- Include 3-6 main branches with 2-3 sub-branches each where possible
- Keep labels short (1-5 words)
- Do not include anything except the Mermaid code block content

Topic: {topic}
Context:
{context}
"""


def _load_db():
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def _format_context(docs) -> str:
    parts: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "")
        page = meta.get("page", "")
        header = f"[source: {source} page: {page}]" if source or page != "" else ""
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def _retrieve_context(query: str, k: int = 8) -> Tuple[List, str]:
    db = _load_db()
    docs = db.similarity_search(query, k=k)
    return docs, _format_context(docs)


def answer_query(question: str, k: int = 4) -> str:
    db = _load_db()
    docs = db.similarity_search(question, k=k)
    context = _format_context(docs)

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm
        resp = chain.invoke({"context": context, "question": question})
        return resp.content

    if not context.strip():
        return "I don't know. Add PDFs/PPTX and ingest first."

    return f"Based on the documents, here is relevant context:\n\n{context}\n\nQuestion: {question}\n\nPlease review the context above to find the answer." 


def summarize_topic(topic: str, k: int = 8) -> str:
    docs, context = _retrieve_context(topic, k=k)
    if not context.strip():
        return "No relevant content found to summarize. Try re-ingesting or rephrasing."

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
        chain = prompt | llm
        resp = chain.invoke({"context": context, "topic": topic})
        return resp.content

    text = context
    text = re.sub(r"\s+", " ", text)
    return f"Summary of {topic}:\n- " + "\n- ".join(text[:1000].split(". ")[:8])


_STOPWORDS = set(
    "the a an and or to of in on for with from by is are was were be as at into over under about across after before during between without within not this that these those it its their his her you your our".split()
)


def _keyword_children(context: str, max_keywords: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", context.lower())
    words = [w for w in words if w not in _STOPWORDS]
    counts = Counter(words)
    common = [w for w, _ in counts.most_common(max_keywords)]
    return [w.replace("-", " ").title() for w in common]


def mindmap_topic(topic: str, k: int = 8) -> str:
    _, context = _retrieve_context(topic, k=k)
    if not context.strip():
        return "mindmap\n  {0}\n    No Content Found".format(topic)

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(MINDMAP_PROMPT)
        chain = prompt | llm
        resp = chain.invoke({"context": context, "topic": topic})
        return resp.content

    children = _keyword_children(context, max_keywords=6)
    return "mindmap\n  {0}\n    {1}".format(topic, "\n    ".join(children) if children else "Overview")


def _get_all_docs_text(source_equals: Optional[str] = None, max_chars: int = 12000) -> Tuple[str, str]:
    db = _load_db()
    try:
        where = {"source": {"$eq": source_equals}} if source_equals else None
        raw = db._collection.get(where=where, include=["documents", "metadatas"])  # type: ignore[attr-defined]
        documents = raw.get("documents", []) or []
        metadatas = raw.get("metadatas", []) or []
        # Flatten potential nested lists
        flat_docs: List[str] = []
        flat_metas: List[dict] = []
        for d in documents:
            if isinstance(d, list):
                flat_docs.extend(d)
            else:
                flat_docs.append(d)
        for m in metadatas:
            if isinstance(m, list):
                flat_metas.extend(m)
            else:
                flat_metas.append(m)
        # Determine a human-friendly title
        sources = [str(m.get("source", "")) for m in flat_metas if m]
        title = os.path.splitext(os.path.basename(sources[0]))[0] if sources else "Chapter"
        # Concatenate with limit
        combined = []
        total = 0
        for t in flat_docs:
            if not t:
                continue
            if total >= max_chars:
                break
            take = t[: max_chars - total]
            combined.append(take)
            total += len(take)
        return "\n\n".join(combined), title
    except Exception:
        docs = db.similarity_search("chapter overview syllabus key points", k=200)
        return _format_context(docs), "Chapter"


def mindmap_chapter(custom_title: Optional[str] = None, source_equals: Optional[str] = None) -> str:
    context, inferred_title = _get_all_docs_text(source_equals=source_equals)
    title = custom_title or inferred_title or "Chapter"
    if not context.strip():
        return "mindmap\n  {0}\n    No Content Found".format(title)

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(MINDMAP_PROMPT)
        chain = prompt | llm
        resp = chain.invoke({"context": context, "topic": title})
        return resp.content

    children = _keyword_children(context, max_keywords=6)
    return "mindmap\n  {0}\n    {1}".format(title, "\n    ".join(children) if children else "Overview")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1].lower() == "summarize":
        print(summarize_topic(" ".join(sys.argv[2:])))
    elif len(sys.argv) > 2 and sys.argv[1].lower() == "mindmap":
        print(mindmap_topic(" ".join(sys.argv[2:])))
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "mindmap_chapter":
        print(mindmap_chapter())
    else:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is in the documents?"
        print(answer_query(q))
