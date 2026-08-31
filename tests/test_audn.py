"""Unit tests for the AUDN (Add / Update / Delete / Noop) classification engine."""

from __future__ import annotations

from mnemo.core.models import AUDNOperation
from mnemo.engine.audn import AUDNClassifier
from mnemo.storage.connection import Database


class TestAUDNPipeline:
    """Test suite for AUDN operations, state transitions, and bitemporal updates."""

    def test_audn_add_new_fact(
        self, in_memory_db: Database, audn_classifier: AUDNClassifier
    ) -> None:
        """New unseen knowledge is classified as ADD and inserted into storage."""
        with in_memory_db.session() as conn:
            op, existing_id = audn_classifier.classify(
                "Brand new discovery about quantum computing", "physics", conn
            )
            assert op == AUDNOperation.ADD
            assert existing_id is None

            fact = audn_classifier.execute_add(
                "Brand new discovery about quantum computing", "physics", conn
            )
            assert fact.text == "Brand new discovery about quantum computing"
            assert fact.is_active is True

            # Verify presence in database
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact.id,)).fetchone()
            assert row is not None
            assert row["text"] == "Brand new discovery about quantum computing"

    def test_audn_noop_duplicate_fact(
        self, in_memory_db: Database, audn_classifier: AUDNClassifier
    ) -> None:
        """Substantially identical facts are classified as NOOP and reinforced."""
        with in_memory_db.session() as conn:
            # 1. Insert original fact
            f_orig = audn_classifier.execute_add(
                "Python is a high level programming language", "programming", conn
            )

            # 2. Incoming duplicate
            op, existing_id = audn_classifier.classify(
                "Python is a high level programming language", "programming", conn
            )
            assert op == AUDNOperation.NOOP
            assert existing_id == f_orig.id

            # 3. Execute NOOP
            audn_classifier.execute_noop(existing_id, conn, boost=0.10)
            row = conn.execute(
                "SELECT access_count, salience FROM facts WHERE id = ?", (f_orig.id,)
            ).fetchone()
            assert row["access_count"] == 1
            assert row["salience"] == 1.0  # Min(1.0, 1.0 + 0.1)

    def test_audn_update_closes_valid_end(
        self, in_memory_db: Database, audn_classifier: AUDNClassifier
    ) -> None:
        """UPDATE operation soft-closes the previous fact and inserts new version."""
        with in_memory_db.session() as conn:
            # 1. Initial fact
            f1 = audn_classifier.execute_add(
                "Project release date is set to October 1", "project", conn
            )

            # 2. Update with new date
            f2 = audn_classifier.execute_update(
                f1.id, "Project release date is postponed to November 15", "project", conn
            )
            assert f2.id != f1.id
            assert f2.text == "Project release date is postponed to November 15"

            # 3. Check old fact validity is closed
            old_row = conn.execute("SELECT valid_end FROM facts WHERE id = ?", (f1.id,)).fetchone()
            assert old_row["valid_end"] is not None

            # 4. Check new fact validity is open
            new_row = conn.execute("SELECT valid_end FROM facts WHERE id = ?", (f2.id,)).fetchone()
            assert new_row["valid_end"] is None

    def test_audn_delete_soft_invalidation(
        self, in_memory_db: Database, audn_classifier: AUDNClassifier
    ) -> None:
        """DELETE operation soft-invalidates fact without physically deleting row."""
        with in_memory_db.session() as conn:
            fact = audn_classifier.execute_add("Fact to be invalidated", "temp", conn)
            audn_classifier.execute_delete(fact.id, conn)

            # Row still exists in DB
            row = conn.execute("SELECT valid_end FROM facts WHERE id = ?", (fact.id,)).fetchone()
            assert row is not None
            assert row["valid_end"] is not None
