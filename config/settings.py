"""
MedGraphRAG — Centralized Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# ============================================================
# Neo4j
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "medgraphrag2024")

# ============================================================
# PostgreSQL
# ============================================================
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "medgraphrag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "medgraphrag2024")

# ============================================================
# MinIO
# ============================================================
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "medgraphrag2024")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "medical-pdfs")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# ============================================================
# Ollama / LLM
# ============================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ============================================================
# LayoutLMv3
# ============================================================
LAYOUTLMV3_MODEL = "microsoft/layoutlmv3-base"
LAYOUTLMV3_MAX_SEQ_LENGTH = 512
LAYOUTLMV3_OVERLAP = 128
IMAGE_DPI = 300

# ============================================================
# NER Labels (BIO format for 10 PII types)
# ============================================================
LABEL_LIST = [
    "O",
    "B-PATIENT_NAME", "I-PATIENT_NAME",
    "B-DOB", "I-DOB",
    "B-AGE", "I-AGE",
    "B-SSN", "I-SSN",
    "B-HOSPITAL_ID", "I-HOSPITAL_ID",
    "B-DOCTOR_NAME", "I-DOCTOR_NAME",
    "B-DOCTOR_ID", "I-DOCTOR_ID",
    "B-HOSPITAL_NAME", "I-HOSPITAL_NAME",
    "B-HOSPITAL_CONTACT", "I-HOSPITAL_CONTACT",
    "B-OTHER_DATE", "I-OTHER_DATE",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# ============================================================
# Training Hyperparameters
# ============================================================
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# ============================================================
# Extraction
# ============================================================
CHUNK_SIZE = 256        # tokens per chunk
CHUNK_OVERLAP = 64      # overlap tokens
FUZZY_THRESHOLD = 0.85  # entity dedup similarity threshold

# ============================================================
# Embedding
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ============================================================
# GraphRAG
# ============================================================
RETRIEVAL_TOP_K = 10
SUBGRAPH_DEPTH = 2
COMMUNITY_SUMMARY_MAX_WORDS = 300

# ============================================================
# Encryption (PII mapping)
# ============================================================
AES_KEY = os.getenv("AES_KEY", "0" * 32)  # 32-byte key — CHANGE IN PRODUCTION

# ============================================================
# Medical Entity & Relation Types
# ============================================================
ENTITY_TYPES = [
    "Diagnosis", "Medication", "Symptom", "Procedure",
    "LabTest", "VitalSign", "ClinicalDate", "Department",
]

RELATION_TYPES = [
    "HAS_DIAGNOSIS", "PRESCRIBED", "EXHIBITS_SYMPTOM",
    "UNDERWENT", "HAS_RESULT", "MEASURED_AT",
    "TREATED_BY", "ADMITTED_TO", "CONTRAINDICATES",
    "INTERACTS_WITH", "CAUSED_BY", "FOLLOWS_UP",
]
