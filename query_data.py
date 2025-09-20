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
from duckduckgo_search import DDGS


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


_ECE_KEYWORDS = set(
    "electronics electronic circuit circuits signal signals systems system control controls communication communications analog digital vlsi fpga verilog vhdl cmos bjt mosfet diode op-amp opamp amplifier adc dac sampling filter filters fourier laplace z-transform transform electromagnetics em electromagnetic radiation microwave antenna antennas embedded microcontroller microprocessor 8051 arm can i2c spi uart plc power electronics drives modulation demodulation information theory coding ldpc ofdm rf ic design pcb sensor sensors transducer transducers robotics robot vision mechatronics instrumentation measurement measurements noise snr cmrr stability pid feedback transfer function state space kcl kvl network theorems rlc resonance transmission line waveguide wave propagation antenna array beamforming s parameters s-parameters smith chart emi emc".split()
)

_NON_ECE_EXCLUDE = set(
    "cricket football soccer ipl worldcup world cup movie politics election celebrity recipe cooking travel tourism horoscope astrology entertainment music song bollywood hollywood box-office weather stock crypto bitcoin nft gaming anime kdrama instagram tiktok memes".split()
)


def _is_ece_query_keywords(text: str) -> bool:
    low = (text or "").lower()
    if any(word in low for word in _NON_ECE_EXCLUDE):
        return False
    return any(word in low for word in _ECE_KEYWORDS)


def _is_ece_query_llm(text: str) -> Optional[bool]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    question = (text or "").strip()
    if not question:
        return None
    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        (
            "You are an intent classifier for Electronics and Communication Engineering (ECE).\n"
            "Classify if the user's query is about ECE academic subjects (signals & systems, EM, circuits, communication, VLSI, control, embedded, etc.).\n"
            "Output EXACTLY one token: ECE or NON-ECE.\n\n"
            "User query: {q}"
        )
    )
    try:
        resp = (prompt | llm).invoke({"q": question})
        out = (resp.content or "").strip().lower()
        if "ece" == out or out.startswith("ece"):
            return True
        if "non-ece" == out or "non ece" in out or out.startswith("non"):
            return False
        # Heuristic if model returns sentence
        if "ece" in out and "non" not in out:
            return True
        if "non" in out:
            return False
        return None
    except Exception:
        return None


def _is_ece_query(text: str) -> bool:
    llm_result = _is_ece_query_llm(text)
    if llm_result is not None:
        return llm_result
    return _is_ece_query_keywords(text)


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


def _retrieve_with_scores(query: str, k: int = 8) -> Tuple[List, float, str]:
    db = _load_db()
    try:
        pairs = db.similarity_search_with_relevance_scores(query, k=k)
        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]
        max_score = max(scores) if scores else 0.0
        return docs, max_score, _format_context(docs)
    except Exception:
        docs = db.similarity_search(query, k=k)
        return docs, 0.0, _format_context(docs)


def _web_search_answer(question: str, max_results: int = 6) -> str:
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(question, max_results=max_results):
                title = r.get("title") or ""
                href = r.get("href") or r.get("url") or ""
                body = r.get("body") or r.get("snippet") or ""
                if title and body and href:
                    results.append({"title": title, "url": href, "snippet": body})
    except Exception:
        pass

    if not os.getenv("OPENAI_API_KEY") or not results:
        # Fallback: return top snippets concatenated
        bullets = [f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results[:3]]
        return "\n".join(bullets) if bullets else "No web results found."

    # Use LLM to synthesize a concise answer from web snippets
    context = "\n\n".join([f"{r['title']}\n{r['snippet']}\n{r['url']}" for r in results])
    llm = ChatOpenAI(model=MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
Summarize the answer to the user's question using ONLY the provided web snippets.
Include 1-2 concise sentences and, if helpful, 1-2 bullet points. Do not fabricate.
Cite sources inline as [n] and then list them at the end as [n] url.

Question: {question}
Snippets:
{snippets}
        """.strip()
    )
    chain = prompt | llm
    resp = chain.invoke({"question": question, "snippets": context})
    return resp.content


def answer_query(question: str, k: int = 4) -> str:
    if not _is_ece_query(question):
        return "I'm built to answer subject/syllabus questions for ECE students."

    docs, max_score, context = _retrieve_with_scores(question, k=max(8, k))

    # If no useful context found, use web fallback with the requested phrasing
    context_missing = (not context.strip()) or (max_score < 0.3)
    if context_missing:
        web_ans = _web_search_answer(question)
        return f"this is not in your context but i'll say you answer:\n\n{web_ans}"

    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model=MODEL, temperature=0)
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | llm
        resp = chain.invoke({"context": context, "question": question})
        return resp.content

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
