-- =========================================================================
-- Mnemo — Bitemporal Memory Store  ·  SQLite 3 DDL
-- =========================================================================
-- Requirements:
--   PRAGMA journal_mode = WAL;
--   PRAGMA foreign_keys = ON;
-- These are set at connection time by connection.py, NOT in this file,
-- because PRAGMA cannot appear inside a transaction / executescript.
-- =========================================================================

-- --------------------------------------------------------------------- --
-- 1. ENTITIES — knowledge-graph nodes
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS entities (
    id              TEXT    PRIMARY KEY,
    name            TEXT    NOT NULL,
    entity_type     TEXT    NOT NULL DEFAULT 'concept',
    properties_json TEXT    NOT NULL DEFAULT '{}',
    salience        REAL    NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed_at REAL   NOT NULL,
    -- bitemporal (epoch seconds)
    valid_start     REAL    NOT NULL,
    valid_end       REAL,
    ingest_start    REAL    NOT NULL,
    ingest_end      REAL
);

-- --------------------------------------------------------------------- --
-- 2. RELATIONS — directed weighted edges
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS relations (
    id              TEXT    PRIMARY KEY,
    source_id       TEXT    NOT NULL,
    target_id       TEXT    NOT NULL,
    relation_type   TEXT    NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    -- bitemporal
    valid_start     REAL    NOT NULL,
    valid_end       REAL,
    ingest_start    REAL    NOT NULL,
    ingest_end      REAL
);

-- --------------------------------------------------------------------- --
-- 3. FACTS — atomic memory records
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS facts (
    id              TEXT    PRIMARY KEY,
    text            TEXT    NOT NULL,
    category        TEXT    NOT NULL DEFAULT 'general',
    salience        REAL    NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    tier            TEXT    NOT NULL DEFAULT 'working',
    last_accessed_at REAL   NOT NULL,
    -- bitemporal
    valid_start     REAL    NOT NULL,
    valid_end       REAL,
    ingest_start    REAL    NOT NULL,
    ingest_end      REAL,
    -- vector blob (stored externally or inline as BLOB)
    embedding_blob  BLOB,
    metadata_json   TEXT    NOT NULL DEFAULT '{}'
);

-- --------------------------------------------------------------------- --
-- 4. DEBT LEDGER — architectural / knowledge debt items
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS debt_ledger (
    id              TEXT    PRIMARY KEY,
    ceiling         TEXT    NOT NULL,
    trigger         TEXT    NOT NULL,
    code_context    TEXT    NOT NULL DEFAULT '',
    created_at      REAL    NOT NULL
);

-- ===================================================================== --
-- 5. FTS5 VIRTUAL TABLE — full-text index over facts
-- ===================================================================== --
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5 (
    id UNINDEXED,
    text,
    category,
    tokenize = 'porter unicode61'
);

-- ===================================================================== --
-- 6. SYNCHRONISATION TRIGGERS  (facts ↔ facts_fts)
-- ===================================================================== --
CREATE TRIGGER IF NOT EXISTS trg_facts_ai
AFTER INSERT ON facts
BEGIN
    INSERT INTO facts_fts (id, text, category)
    SELECT new.id, new.text, new.category
    WHERE new.valid_end IS NULL AND new.ingest_end IS NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_facts_au
AFTER UPDATE ON facts
BEGIN
    DELETE FROM facts_fts WHERE id = old.id;
    INSERT INTO facts_fts (id, text, category)
    SELECT new.id, new.text, new.category
    WHERE new.valid_end IS NULL AND new.ingest_end IS NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_facts_ad
AFTER DELETE ON facts
BEGIN
    DELETE FROM facts_fts WHERE id = old.id;
END;

-- ===================================================================== --
-- 7. B-TREE INDEXES
-- ===================================================================== --
-- Temporal range queries
CREATE INDEX IF NOT EXISTS idx_facts_valid_window
    ON facts (valid_start, valid_end, ingest_end);

CREATE INDEX IF NOT EXISTS idx_facts_tier_salience
    ON facts (tier, salience DESC);

CREATE INDEX IF NOT EXISTS idx_entities_valid_window
    ON entities (valid_start, valid_end, ingest_end);

CREATE INDEX IF NOT EXISTS idx_entities_name_type
    ON entities (name, entity_type);

-- Graph traversal: source → targets, target → sources
CREATE INDEX IF NOT EXISTS idx_relations_source
    ON relations (source_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_relations_target
    ON relations (target_id, relation_type);

CREATE INDEX IF NOT EXISTS idx_relations_valid_window
    ON relations (valid_start, valid_end, ingest_end);

-- Debt ledger by ceiling severity
CREATE INDEX IF NOT EXISTS idx_debt_ceiling
    ON debt_ledger (ceiling, created_at DESC);
