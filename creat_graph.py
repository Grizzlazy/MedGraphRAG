import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from utils import *


def _build_kg_agent() -> KnowledgeGraphAgent:
    """Build KnowledgeGraphAgent backed by the local Ollama/Qwen model."""
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    base_url = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OLLAMA,
        model_type=llm_model,
        model_config_dict={"temperature": 0.0, "max_tokens": 4096},
        url=base_url,
    )
    return KnowledgeGraphAgent(model=model)


def creat_metagraph(args, content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    kg_agent = _build_kg_agent()
    whole_chunk = content

    if args.grained_chunk == True:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j

