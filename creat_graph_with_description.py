"""
融合版图构建函数
使用 nano_graphrag 的 entity_extraction prompt 和解析逻辑
但写入 Neo4j，支持三层架构
"""

import os
import re
import asyncio
from typing import List, Dict
from collections import defaultdict

# Import nano_graphrag components
from nano_graphrag.prompt import PROMPTS
from nano_graphrag._utils import compute_mdhash_id
from nano_graphrag._llm import openai_complete_if_cache

# Import existing components
from camel.loaders import UnstructuredIO
from data_chunk import run_chunk
from utils import get_embedding, str_uuid, add_sum


def clean_str(input_str: str) -> str:
    """清理字符串"""
    if not input_str:
        return ""
    # 移除引号
    input_str = input_str.strip().strip('"').strip("'")
    return input_str


async def extract_entities_with_description(content: str, entity_types: List[str] = None) -> tuple:
    """
    使用 nano_graphrag 的 prompt 提取实体和关系（包含 description）
    
    Args:
        content: 文本内容
        entity_types: 实体类型列表，如果为None则使用默认医学实体类型
    
    Returns:
        (entities, relationships)
        entities: [{'entity_name': ..., 'entity_type': ..., 'description': ...}, ...]
        relationships: [{'src': ..., 'tgt': ..., 'description': ..., 'strength': ...}, ...]
    """
    if entity_types is None:
        # 默认医学实体类型
        entity_types = [
            "Disease", "Symptom", "Treatment", "Medication", "Test", 
            "Anatomy", "Procedure", "Condition", "Measurement", "Hormone",
            "Diagnostic_Criteria", "Clinical_Guideline", "Patient", "Doctor"
        ]
    
    # 使用 nano_graphrag 的 entity_extraction prompt
    prompt_template = PROMPTS["entity_extraction"]
    
    # 准备参数
    entity_types_str = ", ".join(entity_types)
    tuple_delimiter = "<|>"
    record_delimiter = "##"
    completion_delimiter = "<|COMPLETE|>"
    
    # 构建完整 prompt
    prompt = prompt_template.format(
        entity_types=entity_types_str,
        tuple_delimiter=tuple_delimiter,
        record_delimiter=record_delimiter,
        completion_delimiter=completion_delimiter,
        input_text=content[:3000]  # 限制输入长度
    )
    
    # 调用 LLM
    print(f"  [Entity Extraction] 正在提取实体和关系...")
    response = await openai_complete_if_cache(
        model="gpt-4o-mini",
        prompt=prompt,
        system_prompt="You are a helpful assistant that extracts entities and relationships from medical texts."
    )
    
    # 解析响应
    entities = []
    relationships = []
    
    if not response:
        return entities, relationships
    
    # 分割记录
    records = response.split(record_delimiter)
    
    for record in records:
        record = record.strip()
        if not record or completion_delimiter in record:
            continue
        
        # 提取括号内的内容
        match = re.search(r'\((.*?)\)', record)
        if not match:
            continue
        
        record_content = match.group(1)
        attributes = record_content.split(tuple_delimiter)
        
        if len(attributes) < 2:
            continue
        
        record_type = clean_str(attributes[0])
        
        if record_type == "entity" and len(attributes) >= 4:
            # 实体记录: ("entity"<|>entity_name<|>entity_type<|>entity_description)
            entity = {
                'entity_name': clean_str(attributes[1]).upper(),
                'entity_type': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3])
            }
            if entity['entity_name']:
                entities.append(entity)
        
        elif record_type == "relationship" and len(attributes) >= 5:
            # 关系记录: ("relationship"<|>source<|>target<|>description<|>strength)
            relationship = {
                'src': clean_str(attributes[1]).upper(),
                'tgt': clean_str(attributes[2]).upper(),
                'description': clean_str(attributes[3]),
                'strength': clean_str(attributes[4])
            }
            if relationship['src'] and relationship['tgt']:
                relationships.append(relationship)
    
    print(f"  ✅ 提取了 {len(entities)} 个实体，{len(relationships)} 个关系")
    return entities, relationships


def create_neo4j_nodes_and_relationships(n4j, entities: List[Dict], relationships: List[Dict], gid: str):
    """
    将提取的实体和关系写入 Neo4j
    
    Args:
        n4j: Neo4j 连接
        entities: 实体列表
        relationships: 关系列表
        gid: 图ID（用于三层架构）
    """
    print(f"  [Neo4j] 开始写入 {len(entities)} 个实体...")
    
    # 1. 创建实体节点
    for entity in entities:
        entity_name = entity['entity_name']
        entity_type = entity['entity_type']
        description = entity['description']
        
        # 生成 embedding
        embedding_text = f"{entity_name}: {description}" if description else entity_name
        embedding = get_embedding(embedding_text)
        
        # 创建节点的 Cypher 查询
        create_node_query = """
        MERGE (n:`%s` {id: $id, gid: $gid})
        ON CREATE SET 
            n.description = $description,
            n.embedding = $embedding,
            n.source = 'nano_graphrag'
        ON MATCH SET
            n.description = CASE WHEN n.description IS NULL OR n.description = '' 
                                 THEN $description 
                                 ELSE n.description END,
            n.embedding = CASE WHEN n.embedding IS NULL 
                               THEN $embedding 
                               ELSE n.embedding END
        RETURN n
        """ % entity_type
        
        try:
            n4j.query(create_node_query, {
                'id': entity_name,
                'gid': gid,
                'description': description,
                'embedding': embedding
            })
        except Exception as e:
            print(f"  ⚠️  创建节点失败: {entity_name} - {e}")
    
    print(f"  ✅ 实体节点创建完成")
    
    # 2. 创建关系
    print(f"  [Neo4j] 开始创建 {len(relationships)} 个关系...")
    
    for rel in relationships:
        src = rel['src']
        tgt = rel['tgt']
        rel_type = "RELATED_TO"  # 默认关系类型
        
        # 根据描述推断关系类型
        description = rel.get('description', '').lower()
        if 'treat' in description or 'cure' in description:
            rel_type = "TREATS"
        elif 'cause' in description or 'lead' in description:
            rel_type = "CAUSES"
        elif 'diagnose' in description or 'indicate' in description:
            rel_type = "INDICATES"
        elif 'symptom' in description or 'manifest' in description:
            rel_type = "HAS_SYMPTOM"
        
        # 创建关系的 Cypher 查询
        create_rel_query = """
        MATCH (a {id: $src, gid: $gid})
        MATCH (b {id: $tgt, gid: $gid})
        MERGE (a)-[r:`%s`]->(b)
        ON CREATE SET r.description = $description, r.strength = $strength
        RETURN r
        """ % rel_type
        
        try:
            n4j.query(create_rel_query, {
                'src': src,
                'tgt': tgt,
                'gid': gid,
                'description': rel.get('description', ''),
                'strength': rel.get('strength', '')
            })
        except Exception as e:
            print(f"  ⚠️  创建关系失败: {src} -> {tgt} - {e}")
    
    print(f"  ✅ 关系创建完成")


def creat_metagraph_with_description(args, content: str, gid: str, n4j):
    """
    使用 nano_graphrag 的提取逻辑创建知识图谱（包含 description）
    但写入 Neo4j 并支持三层架构（gid）
    
    Args:
        args: 参数
        content: 文本内容
        gid: 图ID（用于三层架构）
        n4j: Neo4j 连接
    
    Returns:
        n4j: 更新后的 Neo4j 连接
    """
    print(f"\n{'='*60}")
    print(f"[图构建] 开始构建知识图谱 (GID: {gid[:8]}...)")
    print(f"{'='*60}")
    
    # 实例化组件
    uio = UnstructuredIO()
    
    # 是否使用细粒度分块
    if args.grained_chunk:
        print("[分块] 使用细粒度分块...")
        content_chunks = run_chunk(content)
    else:
        print("[分块] 使用整体内容...")
        content_chunks = [content]
    
    # 处理每个块
    all_entities = []
    all_relationships = []
    
    for idx, chunk in enumerate(content_chunks, 1):
        print(f"\n[块 {idx}/{len(content_chunks)}] 处理中...")
        
        # 提取实体和关系（异步调用）
        entities, relationships = asyncio.run(
            extract_entities_with_description(chunk)
        )
        
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    print(f"\n[汇总] 总共提取:")
    print(f"  - 实体: {len(all_entities)} 个")
    print(f"  - 关系: {len(all_relationships)} 个")
    
    # 合并重复实体
    print(f"\n[合并] 合并重复实体...")
    entity_dict = {}
    for entity in all_entities:
        name = entity['entity_name']
        if name in entity_dict:
            # 合并描述
            existing_desc = entity_dict[name]['description']
            new_desc = entity['description']
            if new_desc and new_desc not in existing_desc:
                entity_dict[name]['description'] = f"{existing_desc}; {new_desc}"
        else:
            entity_dict[name] = entity
    
    merged_entities = list(entity_dict.values())
    print(f"  ✅ 合并后: {len(merged_entities)} 个实体")
    
    # 写入 Neo4j
    print(f"\n[写入Neo4j] 开始...")
    create_neo4j_nodes_and_relationships(n4j, merged_entities, all_relationships, gid)
    
    # 图内合并（如果启用）
    if args.ingraphmerge:
        print(f"\n[图内合并] 合并相似节点...")
        from utils import merge_similar_nodes
        merge_similar_nodes(n4j, gid)
    
    # 创建 Summary 节点
    print(f"\n[Summary] 创建摘要节点...")
    add_sum(n4j, content, gid)
    
    print(f"\n{'='*60}")
    print(f"[图构建] 完成! (GID: {gid[:8]}...)")
    print(f"{'='*60}\n")
    
    return n4j

