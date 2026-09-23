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
    source_id       TEXT    NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id       TEXT    NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type   TEXT    NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    -- bitemporal
    valid_start     REAL    NOT NULL,
    valid_end       REAL,
    ingest_start    REAL    NOT NULL,
    ingest_end      REAL,
    FOREIGN KEY(source_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY(target_id) REFERENCES entities(id) ON DELETE CASCADE
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
    metadata_json   TEXT    NOT NULL DEFAULT '{}',
    is_stale        INTEGER NOT NULL DEFAULT 0,
    last_accessed_tick INTEGER NOT NULL DEFAULT 0,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    -- provenance and code search tokens
    source_type     TEXT    NOT NULL DEFAULT 'agent',
    source_ref      TEXT,
    confidence      REAL    NOT NULL DEFAULT 1.0,
    search_tokens   TEXT    NOT NULL DEFAULT ''
);

-- --------------------------------------------------------------------- --
-- 4. FACT ENTITY LINKS — association between memory facts and AST entities
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS fact_entity_links (
    fact_id             TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    entity_id           TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    entity_hash_at_link TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (fact_id, entity_id)
);

-- --------------------------------------------------------------------- --
-- 5. DEBT LEDGER — architectural / knowledge debt items
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS debt_ledger (
    id              TEXT    PRIMARY KEY,
    ceiling         TEXT    NOT NULL,
    trigger         TEXT    NOT NULL,
    code_context    TEXT    NOT NULL DEFAULT '',
    created_at      REAL    NOT NULL
);

-- --------------------------------------------------------------------- --
-- 6. PROJECT STATE — global activity ticks and workspace state
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS project_state (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    activity_tick       INTEGER NOT NULL DEFAULT 0,
    updated_at          REAL    NOT NULL,
    embedding_model     TEXT    NOT NULL DEFAULT 'deterministic-hash-64',
    embedding_dimension INTEGER NOT NULL DEFAULT 64
);
INSERT OR IGNORE INTO project_state (id, activity_tick, updated_at, embedding_model, embedding_dimension)
VALUES (1, 0, 0.0, 'deterministic-hash-64', 64);

-- --------------------------------------------------------------------- --
-- 7. AUDIT TOMBSTONES — physical redactions and purged memory registry
-- --------------------------------------------------------------------- --
CREATE TABLE IF NOT EXISTS audit_tombstones (
    id          TEXT    PRIMARY KEY,
    fact_hash   TEXT    NOT NULL,
    reason      TEXT    NOT NULL DEFAULT 'security_redaction',
    purged_at   REAL    NOT NULL
);

-- ===================================================================== --
-- 8. FTS5 VIRTUAL TABLE — full-text index over facts with code tokenchars
-- ===================================================================== --
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5 (
    id UNINDEXED,
    text,
    category,
    tokenize = "unicode61 tokenchars '._'"
);

-- ===================================================================== --
-- 9. SYNCHRONISATION TRIGGERS  (facts ↔ facts_fts)
-- ===================================================================== --
CREATE TRIGGER IF NOT EXISTS trg_facts_ai
AFTER INSERT ON facts
BEGIN
    INSERT INTO facts_fts (id, text, category)
    SELECT new.id, new.text || ' ' || new.search_tokens, new.category
    WHERE new.valid_end IS NULL AND new.ingest_end IS NULL;
END;

CREATE TRIGGER IF NOT EXISTS trg_facts_au
AFTER UPDATE ON facts
BEGIN
    DELETE FROM facts_fts WHERE id = old.id;
    INSERT INTO facts_fts (id, text, category)
    SELECT new.id, new.text || ' ' || new.search_tokens, new.category
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

-- Fact-entity links
CREATE INDEX IF NOT EXISTS idx_fact_entity_links_entity
    ON fact_entity_links (entity_id);

CREATE INDEX IF NOT EXISTS idx_fact_entity_links_fact
    ON fact_entity_links (fact_id);

-- Bitemporal point-in-time index
CREATE INDEX IF NOT EXISTS idx_facts_bitemporal_as_of
    ON facts (valid_start, valid_end, ingest_start, ingest_end);


