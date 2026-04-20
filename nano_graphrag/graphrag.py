import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast


from ._llm import (
    qwen_complete,
    qwen_mini_complete,
    build_local_embedding_func,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_embedding,
)
from ._op import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
    local_query,
    global_query,
)
from ._storage import JsonKVStorage, MilvusLiteStorge, NetworkXStorage
from ._utils import EmbeddingFunc, compute_mdhash_id, limit_async_func_call, logger
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)


@dataclass
class GraphRAG:
    working_dir: str = field(
        default_factory=lambda: f"./nano_graphrag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True

    # parallel chunking (CPU-bound tiktoken, dùng ThreadPool)
    chunk_parallel_workers: int = min(8, (os.cpu_count() or 4))

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        # Fast-eval defaults: force JSON and cap output length.
        default_factory=lambda: {
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 384,
        }
    )
    enable_community_reports: bool = True

    # text embedding
    embedding_func: EmbeddingFunc = field(default_factory=build_local_embedding_func)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    best_model_func: callable = qwen_complete
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 16
    cheap_model_func: callable = qwen_mini_complete
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 16

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = MilvusLiteStorge
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = False

    # extension
    addon_params: dict = field(default_factory=dict)

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"GraphRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=asdict(self)
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(
            self.embedding_func
        )
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )

        self.best_model_func = limit_async_func_call(self.best_model_max_async)(
            partial(self.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.cheap_model_func = limit_async_func_call(self.cheap_model_max_async)(
            partial(self.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert(self, string_or_strings):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "local" and not self.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "local":
            response = await local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        elif param.mode == "global":
            response = await global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, string_or_strings):
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        # ---------- new docs
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            logger.warning(f"All docs are already in the storage")
            return
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        _t0 = time.perf_counter()

        # ---------- chunking (parallel: tiktoken là CPU-bound, dùng ThreadPool)
        def _chunk_one(doc_key_doc):
            doc_key, doc = doc_key_doc
            return {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_key,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=self.chunk_overlap_token_size,
                    max_token_size=self.chunk_token_size,
                    tiktoken_model=self.tiktoken_model_name,
                )
            }

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self.chunk_parallel_workers) as pool:
            chunk_results = await asyncio.gather(
                *[loop.run_in_executor(pool, _chunk_one, item)
                  for item in new_docs.items()]
            )

        inserting_chunks = {}
        for c in chunk_results:
            inserting_chunks.update(c)
        _add_chunk_keys = await self.full_docs.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        _t_chunk = time.perf_counter()
        print(
            f"  [timing] chunking: {_t_chunk - _t0:.2f}s → "
            f"{len(inserting_chunks)} chunks",
            flush=True,
        )
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

        if self.enable_community_reports:
            # TODO: no incremental update for communities now, so just drop all
            await self.community_reports.drop()

        # ---------- extract/summary entity and upsert to graph
        logger.info("[Entity Extraction]...")
        _t_ext0 = time.perf_counter()
        self.chunk_entity_relation_graph = await extract_entities(
            inserting_chunks,
            knwoledge_graph_inst=self.chunk_entity_relation_graph,
            entity_vdb=self.entities_vdb,
            global_config=asdict(self),
        )
        _t_ext1 = time.perf_counter()
        print(
            f"  [timing] entity_extract+merge+vdb: {_t_ext1 - _t_ext0:.2f}s "
            f"(LLM concurrency best={self.best_model_max_async}, "
            f"cheap={self.cheap_model_max_async})",
            flush=True,
        )
        if self.llm_response_cache is not None:
            await self.llm_response_cache.index_done_callback()

        if self.enable_community_reports:
            # ---------- update clusterings of graph
            logger.info("[Community Report]...")
            _t_cl0 = time.perf_counter()
            await self.chunk_entity_relation_graph.clustering(self.graph_cluster_algorithm)
            _t_cl1 = time.perf_counter()
            print(f"  [timing] graph clustering: {_t_cl1 - _t_cl0:.2f}s", flush=True)
            _t_cr0 = time.perf_counter()
            await generate_community_report(
                self.community_reports, self.chunk_entity_relation_graph, asdict(self)
            )
            _t_cr1 = time.perf_counter()
            print(f"  [timing] community LLM reports: {_t_cr1 - _t_cr0:.2f}s", flush=True)
            if self.llm_response_cache is not None:
                await self.llm_response_cache.index_done_callback()

        # ---------- commit upsertings and indexing
        _t_io0 = time.perf_counter()
        await self.full_docs.upsert(new_docs)
        await self.text_chunks.upsert(inserting_chunks)
        await self._insert_done()
        _t_end = time.perf_counter()
        print(f"  [timing] persist (kv+milvus+graphml): {_t_end - _t_io0:.2f}s", flush=True)
        print(
            f"  [timing] === insert TOTAL: {_t_end - _t0:.2f}s "
            f"({len(new_docs)} docs, {len(inserting_chunks)} chunks) ===",
            flush=True,
        )

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.llm_response_cache,
            self.community_reports,
            self.entities_vdb,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def _query_done(self):
        pass
