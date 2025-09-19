from dotenv import load_dotenv
load_dotenv()
from query_data import answer_query, summarize_topic, mindmap_topic, mindmap_chapter


def main():
    print("RAG chatbot. Commands: 'summarize <topic>', 'mindmap <topic>', 'mindmap_chapter', or ask a question. 'exit' to quit.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        low = user.lower()
        if low in {"exit", "quit", "q"}:
            break
        if low.startswith("summarize "):
            topic = user[len("summarize "):].strip()
            print(f"Bot: {summarize_topic(topic)}\n")
            continue
        if low.startswith("mindmap "):
            topic = user[len("mindmap "):].strip()
            print(f"Bot:\n{mindmap_topic(topic)}\n")
            continue
        if low == "mindmap_chapter":
            print(f"Bot:\n{mindmap_chapter()}\n")
            continue
        answer = answer_query(user)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
