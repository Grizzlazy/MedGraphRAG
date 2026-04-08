// ============================================================
// MedGraphRAG — Neo4j Schema Setup
// Run this once after starting Neo4j to create constraints & indexes
// ============================================================

// --- Uniqueness Constraints ---
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT patient_id_unique IF NOT EXISTS
FOR (p:Patient) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT community_id_unique IF NOT EXISTS
FOR (c:Community) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

// --- Property Indexes ---
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_source_idx IF NOT EXISTS FOR (e:Entity) ON (e.source_doc);
CREATE INDEX community_member_count_idx IF NOT EXISTS FOR (c:Community) ON (c.member_count);

// --- Vector Index for Semantic Search ---
// Dimensionality = 384 (all-MiniLM-L6-v2)
CALL db.index.vector.createNodeIndex(
  'entity_embeddings',
  'Entity',
  'embedding',
  384,
  'cosine'
);

// --- Full-Text Index for keyword search ---
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (e:Entity)
ON EACH [e.name, e.description];
