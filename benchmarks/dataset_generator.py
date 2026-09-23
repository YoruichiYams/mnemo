"""Synthetic codebase and knowledge graph dataset generator.

Generates realistic AST entities, relations, and memory facts across 4 challenge groups:
  1. Semantic (conceptual rules without exact code identifiers)
  2. Code Identifiers (camelCase, snake_case, exact symbols)
  3. Graph Dependencies (multi-hop relations across modules)
  4. Bitemporal & Drift (revision chains and stale code drift)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from mnemo.core.models import MemoryTier, SourceType
from mnemo.engine.audn import AUDNClassifier
from mnemo.storage.connection import Database
from mnemo.storage.vector_store import VectorStore, create_embedder


@dataclasses.dataclass
class BenchmarkQuery:
    """A test query with ground truth targets and group category."""

    query: str
    target_fact_ids: list[str]
    group: str
    as_of: float | None = None
    expected_stale: bool = False


@dataclasses.dataclass
class BenchmarkDataset:
    """Generated benchmark environment and query suite."""

    db: Database
    queries: list[BenchmarkQuery]
    facts_count: int
    entities_count: int
    relations_count: int
    scale: str
    timeline: dict[str, float]


def generate_dataset(
    db: Database,
    scale: str = "small",
    base_time: float | None = None,
) -> BenchmarkDataset:
    """Generate synthetic codebase graph and facts for benchmarking."""
    t0 = base_time or (time.time() - 86400 * 30)  # 30 days ago
    t1 = t0 + 86400 * 10
    t2 = t0 + 86400 * 20
    t3 = t0 + 86400 * 25
    now = time.time()

    timeline = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "NOW": now}

    # Scale parameters
    if scale == "medium":
        service_count = 100
    elif scale == "large":
        service_count = 400
    else:  # small
        service_count = 15

    vs = VectorStore(create_embedder())
    audn = AUDNClassifier(vs)

    queries: list[BenchmarkQuery] = []

    with db.session() as conn:
        # -------------------------------------------------------------
        # 1. Generate AST Entities & Architecture Graph
        # -------------------------------------------------------------
        entities_data: list[dict[str, Any]] = []
        relations_data: list[dict[str, Any]] = []

        services = [
            ("OrderWorkflowService", "src/services/order.py"),
            ("PaymentGatewayService", "src/services/payment.py"),
            ("AuthenticationService", "src/services/auth.py"),
            ("UserAccountService", "src/services/user.py"),
            ("InventorySyncService", "src/services/inventory.py"),
            ("NotificationDispatchService", "src/services/notification.py"),
            ("BillingReportService", "src/services/billing.py"),
            ("MetricsCollectorService", "src/services/metrics.py"),
            ("AuditLogService", "src/services/audit.py"),
            ("TelemetryAggregatorService", "src/services/telemetry.py"),
            ("SecurityPolicyService", "src/services/security.py"),
            ("RateLimiterService", "src/services/ratelimit.py"),
            ("CacheInvalidationService", "src/services/cache.py"),
            ("WebhookDeliveryService", "src/services/webhook.py"),
            ("DatabaseMigrationService", "src/services/db_migrator.py"),
        ]

        # Expand if scale requires
        if service_count > len(services):
            for i in range(len(services), service_count):
                services.append((f"SyntheticModuleService{i}", f"src/modules/module_{i}.py"))

        for s_name, _s_file in services:
            s_id = f"ent_{s_name}"
            entities_data.append(
                {
                    "id": s_id,
                    "name": s_name,
                    "entity_type": "class",
                    "properties_json": "{}",
                    "salience": 1.0,
                    "access_count": 0,
                    "last_accessed_at": t0,
                    "valid_start": t0,
                    "valid_end": None,
                    "ingest_start": t0,
                    "ingest_end": None,
                }
            )

            # Sub-methods for each service
            methods = [
                f"{s_name}.initialize_handler",
                f"{s_name}.execute_transaction",
                f"{s_name}.validate_payload",
                f"{s_name}.rollback_changes",
            ]
            for m_name in methods:
                m_id = f"ent_{m_name}"
                entities_data.append(
                    {
                        "id": m_id,
                        "name": m_name,
                        "entity_type": "function",
                        "properties_json": "{}",
                        "salience": 1.0,
                        "access_count": 0,
                        "last_accessed_at": t0,
                        "valid_start": t0,
                        "valid_end": None,
                        "ingest_start": t0,
                        "ingest_end": None,
                    }
                )
                relations_data.append(
                    {
                        "id": f"rel_{s_id}_{m_id}",
                        "source_id": s_id,
                        "target_id": m_id,
                        "relation_type": "defines",
                        "weight": 1.0,
                        "valid_start": t0,
                        "valid_end": None,
                        "ingest_start": t0,
                        "ingest_end": None,
                    }
                )

        # Architectural dependencies (Service A -> Service B -> Repository)
        # OrderWorkflowService -> PaymentGatewayService -> PostgresPaymentRepository
        repo_id = "ent_PostgresPaymentRepository"
        entities_data.append(
            {
                "id": repo_id,
                "name": "PostgresPaymentRepository",
                "entity_type": "class",
                "properties_json": "{}",
                "salience": 1.0,
                "access_count": 0,
                "last_accessed_at": t0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )

        relations_data.append(
            {
                "id": "rel_order_payment",
                "source_id": "ent_OrderWorkflowService",
                "target_id": "ent_PaymentGatewayService",
                "relation_type": "depends_on",
                "weight": 1.0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )
        relations_data.append(
            {
                "id": "rel_payment_repo",
                "source_id": "ent_PaymentGatewayService",
                "target_id": repo_id,
                "relation_type": "depends_on",
                "weight": 1.0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )

        # AuthenticationService -> UserAccountService -> PostgresUserRepository
        user_repo_id = "ent_PostgresUserRepository"
        entities_data.append(
            {
                "id": user_repo_id,
                "name": "PostgresUserRepository",
                "entity_type": "class",
                "properties_json": "{}",
                "salience": 1.0,
                "access_count": 0,
                "last_accessed_at": t0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )
        relations_data.append(
            {
                "id": "rel_auth_user",
                "source_id": "ent_AuthenticationService",
                "target_id": "ent_UserAccountService",
                "relation_type": "depends_on",
                "weight": 1.0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )
        relations_data.append(
            {
                "id": "rel_user_repo",
                "source_id": "ent_UserAccountService",
                "target_id": user_repo_id,
                "relation_type": "depends_on",
                "weight": 1.0,
                "valid_start": t0,
                "valid_end": None,
                "ingest_start": t0,
                "ingest_end": None,
            }
        )

        # Bulk insert entities and relations
        for e in entities_data:
            conn.execute(
                "INSERT OR REPLACE INTO entities "
                "(id, name, entity_type, properties_json, salience, access_count, "
                " last_accessed_at, valid_start, valid_end, ingest_start, ingest_end) "
                "VALUES (:id, :name, :entity_type, :properties_json, :salience, :access_count, "
                " :last_accessed_at, :valid_start, :valid_end, :ingest_start, :ingest_end)",
                e,
            )
        for r in relations_data:
            conn.execute(
                "INSERT OR REPLACE INTO relations "
                "(id, source_id, target_id, relation_type, weight, "
                " valid_start, valid_end, ingest_start, ingest_end) "
                "VALUES (:id, :source_id, :target_id, :relation_type, :weight, "
                " :valid_start, :valid_end, :ingest_start, :ingest_end)",
                r,
            )

        # -------------------------------------------------------------
        # 2. Fact Group 1: Semantic Queries (Conceptual Rules)
        # -------------------------------------------------------------
        semantic_facts = [
            (
                "All financial ledger mutations require idempotent distributed locks via Redis Redlock to avoid double charge",
                "architecture",
                "How do we prevent double spending and charge collisions during transactions?",
            ),
            (
                "Database connection timeouts must be set to thirty seconds with jittered exponential backoff to handle connection pool exhaustion",
                "infra",
                "What is the recommended timeout and retry strategy for overloaded database pools?",
            ),
            (
                "Zero trust security mandates cryptographic mutual TLS for all east-west microservice traffic in the cluster",
                "security",
                "What protocol secures internal communication between backend microservices?",
            ),
            (
                "All external API requests must include correlation ID headers in logs to facilitate distributed tracing",
                "observability",
                "How do we trace incoming HTTP calls across multiple downstream services?",
            ),
            (
                "User password hashes must use Argon2id with minimum memory cost of 64MB and four parallel threads",
                "security",
                "What hashing algorithm and memory parameters protect stored user passwords?",
            ),
            (
                "Read-after-write consistency for updated profile states is enforced via read replica pin routing for sixty seconds",
                "database",
                "How do we prevent stale profile reads immediately after an update?",
            ),
            (
                "Kafka event consumers must commit offsets only after transactional database flush completes successfully",
                "messaging",
                "When is it safe for event listeners to commit message offsets in message queues?",
            ),
            (
                "Critical customer alerts must bypass standard notification batch queues and route directly through high priority SMS gateways",
                "notification",
                "How are urgent emergency alerts handled compared to regular emails?",
            ),
            (
                "Rate limiter buckets replenish tokens every hundred milliseconds with max burst capacity restricted to double nominal rate",
                "traffic",
                "What is the token bucket replenishment rate and burst limit for traffic control?",
            ),
            (
                "Audit logs must be written to append-only immutable storage with daily SHA256 integrity verification seals",
                "compliance",
                "How do we guarantee that compliance audit logs cannot be tampered with or deleted?",
            ),
            (
                "Cache eviction policy uses adaptive LFU with salience weighting to retain frequently accessed entity schemas",
                "cache",
                "What cache replacement policy keeps the most valuable schemas in memory?",
            ),
            (
                "Webhook deliveries require exponential backoff with dead letter queue routing after five consecutive failed attempts",
                "webhooks",
                "What is the retry threshold and fallback destination for failing webhooks?",
            ),
            (
                "Database schema migrations must be backward compatible and execute in non-blocking lock-free mode without exclusive table locks",
                "migrations",
                "What constraints are placed on database migrations to ensure zero downtime?",
            ),
            (
                "Service health checks must distinguish between readiness probe and liveness probe to prevent cascading restart loops",
                "k8s",
                "How do health probes avoid triggering cascading pod restarts in the cluster?",
            ),
            (
                "Tenant data isolation in multitenant queries is strictly enforced via PostgreSQL row level security policies",
                "security",
                "How is multitenant data segregation enforced at the database level?",
            ),
        ]

        for text, cat, query_text in semantic_facts:
            fact = audn.execute_add(
                text,
                cat,
                conn,
                tier=MemoryTier.WORKING,
                source_type=SourceType.DOCUMENTATION.value,
                confidence=1.0,
            )
            queries.append(
                BenchmarkQuery(
                    query=query_text,
                    target_fact_ids=[fact.id],
                    group="semantic",
                )
            )

        # -------------------------------------------------------------
        # 3. Fact Group 2: Code Identifiers (Exact Symbols)
        # -------------------------------------------------------------
        code_id_facts = [
            (
                "OrderWorkflowService.execute_transaction validates order items against inventory before reserving funds",
                "code",
                "OrderWorkflowService.execute_transaction inventory validation",
                "OrderWorkflowService",
            ),
            (
                "PaymentGatewayService.execute_transaction dispatches payment payload to Stripe and returns PaymentIntent status",
                "code",
                "PaymentGatewayService.execute_transaction payment payload dispatch",
                "PaymentGatewayService",
            ),
            (
                "AuthenticationService.initialize_handler configures OAuth2 PKCE state verification cookies",
                "code",
                "AuthenticationService.initialize_handler OAuth2 PKCE",
                "AuthenticationService",
            ),
            (
                "UserAccountService.rollback_changes restores previous user preferences from snapshot backup",
                "code",
                "UserAccountService.rollback_changes snapshot backup",
                "UserAccountService",
            ),
            (
                "InventorySyncService.execute_transaction updates warehouse stock quantities using pessimistic row locks",
                "code",
                "InventorySyncService.execute_transaction pessimistic row locks",
                "InventorySyncService",
            ),
            (
                "NotificationDispatchService.validate_payload checks for required recipient phone number in E164 format",
                "code",
                "NotificationDispatchService.validate_payload E164 format",
                "NotificationDispatchService",
            ),
            (
                "BillingReportService.initialize_handler sets up monthly aggregated invoice generation cron job",
                "code",
                "BillingReportService.initialize_handler invoice cron job",
                "BillingReportService",
            ),
            (
                "MetricsCollectorService.execute_transaction flushes Prometheus counters and histograms every ten seconds",
                "code",
                "MetricsCollectorService.execute_transaction Prometheus counters",
                "MetricsCollectorService",
            ),
            (
                "AuditLogService.rollback_changes writes security incident audit record upon transaction rollback",
                "code",
                "AuditLogService.rollback_changes security incident audit",
                "AuditLogService",
            ),
            (
                "RateLimiterService.validate_payload inspects X-Forwarded-For header to extract client IP subnet",
                "code",
                "RateLimiterService.validate_payload X-Forwarded-For client IP",
                "RateLimiterService",
            ),
            (
                "CacheInvalidationService.execute_transaction purges Redis keys matching tag pattern user_session_*",
                "code",
                "CacheInvalidationService.execute_transaction user_session_* Redis keys",
                "CacheInvalidationService",
            ),
            (
                "WebhookDeliveryService.validate_payload signs outbound JSON body with HMAC SHA256 secret key",
                "code",
                "WebhookDeliveryService.validate_payload HMAC SHA256 secret key",
                "WebhookDeliveryService",
            ),
            (
                "DatabaseMigrationService.execute_transaction applies flyway versioned DDL scripts within explicit transaction",
                "code",
                "DatabaseMigrationService.execute_transaction flyway versioned DDL",
                "DatabaseMigrationService",
            ),
            (
                "SecurityPolicyService.initialize_handler loads CORS origin whitelist from environment configuration",
                "code",
                "SecurityPolicyService.initialize_handler CORS origin whitelist",
                "SecurityPolicyService",
            ),
            (
                "TelemetryAggregatorService.rollback_changes drops unparsed OpenTelemetry spans if buffer overflows",
                "code",
                "TelemetryAggregatorService.rollback_changes unparsed OpenTelemetry spans",
                "TelemetryAggregatorService",
            ),
        ]

        for text, cat, query_text, ent_name in code_id_facts:
            fact = audn.execute_add(
                text,
                cat,
                conn,
                tier=MemoryTier.WORKING,
                source_type=SourceType.GIT_COMMIT.value,
                confidence=1.0,
            )
            # Explicit link
            ent_id = f"ent_{ent_name}"
            conn.execute(
                "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
                "VALUES (?, ?, 'init_hash', ?)",
                (fact.id, ent_id, str(now)),
            )
            queries.append(
                BenchmarkQuery(
                    query=query_text,
                    target_fact_ids=[fact.id],
                    group="code_id",
                )
            )

        # -------------------------------------------------------------
        # 4. Fact Group 3: Graph Dependencies (Multi-hop Architectural Traversal)
        # -------------------------------------------------------------
        # Fact linked to PostgresPaymentRepository: query asks about OrderWorkflowService
        f_dep1 = audn.execute_add(
            "PostgresPaymentRepository requires serializable transaction isolation level to ensure ledger balances never diverge",
            "database",
            conn,
            tier=MemoryTier.CORE,
            source_type=SourceType.HUMAN_DEVELOPER.value,
            confidence=1.0,
        )
        conn.execute(
            "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
            "VALUES (?, 'ent_PostgresPaymentRepository', 'hash_repo1', ?)",
            (f_dep1.id, str(now)),
        )
        queries.append(
            BenchmarkQuery(
                query="What transaction isolation level is required when OrderWorkflowService processes orders?",
                target_fact_ids=[f_dep1.id],
                group="graph_dep",
            )
        )

        # Fact linked to PostgresUserRepository: query asks about AuthenticationService
        f_dep2 = audn.execute_add(
            "PostgresUserRepository requires read replica connection pool with failover replica health verification before auth",
            "database",
            conn,
            tier=MemoryTier.CORE,
            source_type=SourceType.HUMAN_DEVELOPER.value,
            confidence=1.0,
        )
        conn.execute(
            "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
            "VALUES (?, 'ent_PostgresUserRepository', 'hash_repo2', ?)",
            (f_dep2.id, str(now)),
        )
        queries.append(
            BenchmarkQuery(
                query="What database connection requirements apply to AuthenticationService during user credential lookup?",
                target_fact_ids=[f_dep2.id],
                group="graph_dep",
            )
        )

        # Fact linked to PaymentGatewayService: query asks about OrderWorkflowService
        f_dep3 = audn.execute_add(
            "PaymentGatewayService mandates Stripe webhook signature verification using secret STRIPE_SIGNING_KEY",
            "security",
            conn,
            tier=MemoryTier.WORKING,
            source_type=SourceType.GIT_COMMIT.value,
            confidence=1.0,
        )
        conn.execute(
            "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
            "VALUES (?, 'ent_PaymentGatewayService', 'hash_stripe', ?)",
            (f_dep3.id, str(now)),
        )
        queries.append(
            BenchmarkQuery(
                query="How does OrderWorkflowService authenticate inbound third-party payment callbacks?",
                target_fact_ids=[f_dep3.id],
                group="graph_dep",
            )
        )

        # Additional graph dependency facts to satisfy 50+ total
        for i in range(1, 18):
            s_source = services[i % len(services)][0]
            s_target = services[(i + 1) % len(services)][0]
            rel_id = f"rel_dep_{s_source}_{s_target}_{i}"
            conn.execute(
                "INSERT OR REPLACE INTO relations (id, source_id, target_id, relation_type, weight, valid_start, valid_end, ingest_start, ingest_end) "
                "VALUES (?, ?, ?, 'calls', 1.0, ?, NULL, ?, NULL)",
                (rel_id, f"ent_{s_source}", f"ent_{s_target}", t0, t0),
            )
            fact_dep = audn.execute_add(
                f"{s_target} requires encrypted gRPC payloads with payload signature from upstream {s_source}",
                "architecture",
                conn,
                tier=MemoryTier.WORKING,
                source_type=SourceType.GIT_COMMIT.value,
                confidence=1.0,
            )
            conn.execute(
                "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
                "VALUES (?, ?, 'hash_link', ?)",
                (fact_dep.id, f"ent_{s_target}", str(now)),
            )
            queries.append(
                BenchmarkQuery(
                    query=f"What payload security is expected when {s_source} interacts with downstream services?",
                    target_fact_ids=[fact_dep.id],
                    group="graph_dep",
                )
            )

        # -------------------------------------------------------------
        # 5. Fact Group 4: Bitemporal & Drift Benchmark
        # -------------------------------------------------------------
        # Lifecycle: ADD (T0) -> CORRECT (T1) -> INVALIDATE (T2)
        # Version 1 (Valid at T0, corrected at T1)
        f_v1_text = "API rate limit is 100 requests per minute per IP address"
        f_v1 = audn.execute_add(
            f_v1_text,
            "ratelimit",
            conn,
            tier=MemoryTier.WORKING,
            source_type=SourceType.HUMAN_DEVELOPER.value,
            confidence=1.0,
        )
        # Manually backdate v1 to T0
        conn.execute(
            "UPDATE facts SET valid_start = ?, ingest_start = ?, valid_end = NULL, ingest_end = ? WHERE id = ?",
            (t0, t0, t1, f_v1.id),
        )

        # Version 2 (Valid at T1, corrected from v1, valid until T2)
        f_v2_text = "API rate limit is 500 requests per minute per IP address"
        f_v2 = audn.execute_add(
            f_v2_text,
            "ratelimit",
            conn,
            tier=MemoryTier.WORKING,
            source_type=SourceType.HUMAN_DEVELOPER.value,
            confidence=1.0,
        )
        conn.execute(
            "UPDATE facts SET valid_start = ?, ingest_start = ?, valid_end = ?, ingest_end = NULL WHERE id = ?",
            (t0, t1, t2, f_v2.id),
        )

        # Version 3 (Active from T2 onward)
        f_v3_text = "API rate limit is dynamic based on customer SLA tier with minimum 1000 requests per minute"
        f_v3 = audn.execute_add(
            f_v3_text,
            "ratelimit",
            conn,
            tier=MemoryTier.WORKING,
            source_type=SourceType.HUMAN_DEVELOPER.value,
            confidence=1.0,
        )
        conn.execute(
            "UPDATE facts SET valid_start = ?, ingest_start = ?, valid_end = NULL, ingest_end = NULL WHERE id = ?",
            (t2, t2, f_v3.id),
        )

        # Bitemporal queries at specific points in time
        queries.append(
            BenchmarkQuery(
                query="What was the API rate limit per IP address?",
                target_fact_ids=[f_v1.id],
                group="bitemporal",
                as_of=t0 + 3600,  # 1 hour after T0
            )
        )
        queries.append(
            BenchmarkQuery(
                query="What was the API rate limit per IP address?",
                target_fact_ids=[f_v2.id],
                group="bitemporal",
                as_of=t1 + 3600,  # 1 hour after T1
            )
        )
        queries.append(
            BenchmarkQuery(
                query="What is the current API rate limit per IP address?",
                target_fact_ids=[f_v3.id],
                group="bitemporal",
                as_of=now,
            )
        )

        # AST Drift: fact linked to entity whose content hash changed
        drift_entity_id = "ent_SecurityPolicyService"
        f_drift = audn.execute_add(
            "SecurityPolicyService.validate_payload enforces strict Content-Security-Policy with nonce generation",
            "security",
            conn,
            tier=MemoryTier.WORKING,
            source_type=SourceType.GIT_COMMIT.value,
            confidence=1.0,
        )
        # Link with original hash
        conn.execute(
            "INSERT OR REPLACE INTO fact_entity_links (fact_id, entity_id, entity_hash_at_link, created_at) "
            "VALUES (?, ?, 'original_hash_123', ?)",
            (f_drift.id, drift_entity_id, str(t0)),
        )
        # Simulate drift by setting is_stale = 1 on fact
        conn.execute("UPDATE facts SET is_stale = 1 WHERE id = ?", (f_drift.id,))

        queries.append(
            BenchmarkQuery(
                query="SecurityPolicyService.validate_payload Content-Security-Policy",
                target_fact_ids=[f_drift.id],
                group="bitemporal",
                expected_stale=True,
            )
        )

        # Additional background noise facts to match requested scale
        current_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        needed_facts = (500 if scale == "small" else (5000 if scale == "medium" else 20000))
        remaining = max(0, needed_facts - current_facts)

        if remaining > 0:
            bulk_inserts = []
            for i in range(remaining):
                fid = f"noise_fact_{scale}_{i}"
                txt = f"Synthetic operational background noise fact {i} relating to server cluster {i % 10} metrics"
                cat = "telemetry"
                now_ts = now - (i * 60)
                bulk_inserts.append(
                    (fid, txt, cat, 0.5, 0, "archived" if i % 3 == 0 else "working", now_ts, 0, 0, now_ts, None, now_ts, None, None, "{}", 0, "agent", None, 0.9, "")
                )
            conn.executemany(
                "INSERT INTO facts "
                "(id, text, category, salience, access_count, tier, last_accessed_at, "
                " last_accessed_tick, reinforcement_count, valid_start, valid_end, "
                " ingest_start, ingest_end, embedding_blob, metadata_json, is_stale, "
                " source_type, source_ref, confidence, search_tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                bulk_inserts,
            )
            # Synchronize FTS5 for noise facts
            conn.execute(
                "INSERT INTO facts_fts (id, text, category) "
                "SELECT id, text || ' ' || search_tokens, category FROM facts "
                "WHERE id LIKE 'noise_fact_%'"
            )

        f_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        e_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        r_count = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]

    return BenchmarkDataset(
        db=db,
        queries=queries,
        facts_count=f_count,
        entities_count=e_count,
        relations_count=r_count,
        scale=scale,
        timeline=timeline,
    )
