import json
from typing import List
import os
from agentic_chunker import AgenticChunker
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_propositions(text, runnable):
    runnable_output = runnable.invoke({
    	"input": text
    }).content

    try:
        extracted = json.loads(runnable_output)
        sentences = extracted.get("sentences", [])
        return [s for s in sentences if isinstance(s, str) and s.strip()]
    except Exception:
        return [line.strip("- ").strip() for line in runnable_output.splitlines() if line.strip()]

def run_chunk(essay):

    obj = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract atomic propositions from the input text. "
                "Return strict JSON in this format only: "
                "{{\"sentences\": [\"...\", \"...\"]}}.",
            ),
            ("user", "{input}"),
        ]
    )
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen3.5:9b"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        openai_api_base=os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1"),
    )
    runnable = obj | llm

    paragraphs = essay.split("\n\n")

    essay_propositions = []

    for i, para in enumerate(paragraphs):
        propositions = get_propositions(para, runnable)
        
        essay_propositions.extend(propositions)
        print (f"Done with {i}")

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks
    print(chunks)