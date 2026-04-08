"""
MedGraphRAG — PostgreSQL Client
Manages PII reverse-mapping storage with AES-256 encryption.
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from loguru import logger
import base64, os

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD, AES_KEY,
)


class PostgresClient:
    """Handles PII mapping persistence with AES-256 encryption."""

    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        self.conn.autocommit = True
        self._key = AES_KEY.encode("utf-8")[:32].ljust(32, b"\0")
        self._init_tables()
        logger.info("PostgreSQL client initialized")

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------
    def _init_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id              SERIAL PRIMARY KEY,
                    doc_id          VARCHAR(255) UNIQUE NOT NULL,
                    filename        VARCHAR(512),
                    minio_path      VARCHAR(1024),
                    created_at      TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pii_mappings (
                    id              SERIAL PRIMARY KEY,
                    doc_id          VARCHAR(255) NOT NULL REFERENCES documents(doc_id),
                    placeholder     VARCHAR(64) NOT NULL,
                    encrypted_data  TEXT NOT NULL,
                    entity_type     VARCHAR(64),
                    created_at      TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pii_doc ON pii_mappings(doc_id);
            """)

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------
    def _encrypt(self, plaintext: str) -> str:
        """AES-256-CBC encryption, returns base64 string."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        # PKCS7 padding
        pad_len = 16 - (len(plaintext.encode()) % 16)
        padded = plaintext.encode() + bytes([pad_len] * pad_len)
        ct = encryptor.update(padded) + encryptor.finalize()
        return base64.b64encode(iv + ct).decode()

    def _decrypt(self, ciphertext_b64: str) -> str:
        """AES-256-CBC decryption."""
        raw = base64.b64decode(ciphertext_b64)
        iv, ct = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(self._key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ct) + decryptor.finalize()
        pad_len = padded[-1]
        return padded[:-pad_len].decode()

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------
    def register_document(self, doc_id: str, filename: str, minio_path: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (doc_id, filename, minio_path)
                VALUES (%s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET filename = EXCLUDED.filename
            """, (doc_id, filename, minio_path))

    def get_document(self, doc_id: str) -> dict | None:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # PII Mapping CRUD
    # ------------------------------------------------------------------
    def store_pii_mapping(self, doc_id: str, mapping: dict[str, dict]) -> None:
        """
        Store PII reverse-mapping with encryption.

        mapping: { "[NAME_001]": {"original": "John Doe", "type": "PATIENT_NAME", ...}, ... }
        """
        with self.conn.cursor() as cur:
            for placeholder, data in mapping.items():
                encrypted = self._encrypt(json.dumps(data))
                cur.execute("""
                    INSERT INTO pii_mappings (doc_id, placeholder, encrypted_data, entity_type)
                    VALUES (%s, %s, %s, %s)
                """, (doc_id, placeholder, encrypted, data.get("type", "")))
        logger.info(f"Stored {len(mapping)} PII mappings for doc {doc_id}")

    def get_pii_mapping(self, doc_id: str) -> dict[str, dict]:
        """Retrieve and decrypt all PII mappings for a document."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT placeholder, encrypted_data FROM pii_mappings WHERE doc_id = %s",
                (doc_id,),
            )
            mapping = {}
            for row in cur.fetchall():
                decrypted = json.loads(self._decrypt(row["encrypted_data"]))
                mapping[row["placeholder"]] = decrypted
            return mapping

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        self.conn.close()
        logger.info("PostgreSQL connection closed")
