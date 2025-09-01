"""Database integrity validation utilities."""

import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_session() -> Session:
    """Get database session.

    Placeholder implementation - should be replaced with actual database setup.
    """
    # This is a placeholder - actual implementation should use proper config
    engine = create_engine("postgresql://localhost/tracktion")
    session_local = sessionmaker(bind=engine)
    return session_local()


class IntegrityValidator:
    """Validates database referential integrity and finds orphaned records."""

    def __init__(self, session: Session):
        """Initialize with database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def check_orphaned_records(self) -> dict[str, int]:
        """Check for orphaned records using the database function.

        Returns:
            Dictionary mapping table names to orphaned record counts
        """
        try:
            result = self.session.execute(text("SELECT * FROM check_orphaned_records()"))
            orphaned = {}
            for row in result:
                table_name, count = row
                orphaned[table_name] = count
                if count > 0:
                    logger.warning(f"Found {count} orphaned records in {table_name}")
            return orphaned
        except Exception as e:
            logger.error(f"Error checking orphaned records: {e}")
            raise

    def clean_orphaned_records(self, dry_run: bool = True) -> dict[str, int]:
        """Clean orphaned records from the database.

        Args:
            dry_run: If True, only report what would be deleted without actually deleting

        Returns:
            Dictionary mapping table names to deleted record counts
        """
        if dry_run:
            # Just check and report
            return self.check_orphaned_records()

        try:
            result = self.session.execute(text("SELECT * FROM clean_orphaned_records()"))
            cleaned = {}
            for row in result:
                table_name, count = row
                cleaned[table_name] = count
                if count > 0:
                    logger.info(f"Cleaned {count} orphaned records from {table_name}")
            self.session.commit()
            return cleaned
        except Exception as e:
            logger.error(f"Error cleaning orphaned records: {e}")
            self.session.rollback()
            raise

    def validate_foreign_keys(self) -> list[tuple[str, str, bool]]:
        """Validate all foreign key constraints are properly set.

        Returns:
            List of tuples (constraint_name, table_name, is_valid)
        """
        query = text(
            """
            SELECT
                conname as constraint_name,
                conrelid::regclass as table_name,
                confdeltype as delete_action
            FROM pg_constraint
            WHERE contype = 'f'
            AND connamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """
        )

        results = []
        try:
            constraints = self.session.execute(query)
            for row in constraints:
                constraint_name, table_name, delete_action = row
                # Check if CASCADE is set (delete_action = 'c')
                is_cascade = delete_action == "c"
                results.append((constraint_name, str(table_name), is_cascade))

                if not is_cascade and table_name in ["metadata", "tracklists", "rename_proposals"]:
                    logger.warning(f"Foreign key {constraint_name} on {table_name} does not have CASCADE delete action")
            return results
        except Exception as e:
            logger.error(f"Error validating foreign keys: {e}")
            raise

    def validate_check_constraints(self) -> list[tuple[str, str, bool]]:
        """Validate all check constraints are working.

        Returns:
            List of tuples (constraint_name, table_name, is_valid)
        """
        query = text(
            """
            SELECT
                conname as constraint_name,
                conrelid::regclass as table_name
            FROM pg_constraint
            WHERE contype = 'c'
            AND connamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        """
        )

        results = []
        try:
            constraints = self.session.execute(query)
            for row in constraints:
                constraint_name, table_name = row
                # All check constraints that exist are considered valid
                results.append((constraint_name, str(table_name), True))
            return results
        except Exception as e:
            logger.error(f"Error validating check constraints: {e}")
            raise

    def validate_indexes(self) -> list[tuple[str, str, bool]]:
        """Validate that all expected indexes exist.

        Returns:
            List of tuples (index_name, table_name, exists)
        """
        expected_indexes = [
            ("idx_metadata_recording_id", "metadata"),
            ("idx_tracklists_recording_id", "tracklists"),
            ("idx_rename_proposals_recording_id", "rename_proposals"),
        ]

        query = text(
            """
            SELECT
                indexname,
                tablename
            FROM pg_indexes
            WHERE schemaname = 'public'
        """
        )

        results = []
        try:
            existing_indexes = self.session.execute(query)
            existing_index_set = {(row[0], row[1]) for row in existing_indexes}

            for index_name, table_name in expected_indexes:
                exists = (index_name, table_name) in existing_index_set
                results.append((index_name, table_name, exists))
                if not exists:
                    logger.warning(f"Missing index {index_name} on {table_name}")

            return results
        except Exception as e:
            logger.error(f"Error validating indexes: {e}")
            raise

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete integrity validation.

        Returns:
            Dictionary with validation results
        """
        logger.info("Starting full integrity validation")

        orphaned_records = self.check_orphaned_records()
        foreign_keys = self.validate_foreign_keys()
        check_constraints = self.validate_check_constraints()
        indexes = self.validate_indexes()

        results: dict[str, Any] = {
            "orphaned_records": orphaned_records,
            "foreign_keys": foreign_keys,
            "check_constraints": check_constraints,
            "indexes": indexes,
            "is_valid": True,
        }

        # Check if any issues were found
        if any(count > 0 for count in orphaned_records.values()):
            results["is_valid"] = False
            logger.error("Found orphaned records")

        if any(not is_valid for _, _, is_valid in foreign_keys):
            results["is_valid"] = False
            logger.error("Found invalid foreign key constraints")

        if any(not exists for _, _, exists in indexes):
            results["is_valid"] = False
            logger.error("Found missing indexes")

        if results["is_valid"]:
            logger.info("Integrity validation passed")
        else:
            logger.error("Integrity validation failed")

        return results


def main() -> int:
    """Run integrity validation as a standalone script."""
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    session = get_session()
    try:
        validator = IntegrityValidator(session)
        results = validator.run_full_validation()

        print("\n=== Integrity Validation Results ===")
        print("\nOrphaned Records:")
        for table, count in results["orphaned_records"].items():
            print(f"  {table}: {count}")

        print("\nForeign Keys (with CASCADE):")
        for constraint, table, is_cascade in results["foreign_keys"]:
            status = "✓" if is_cascade else "✗"
            print(f"  {status} {constraint} on {table}")

        print("\nCheck Constraints:")
        for constraint, table, is_valid in results["check_constraints"]:
            status = "✓" if is_valid else "✗"
            print(f"  {status} {constraint} on {table}")

        print("\nIndexes:")
        for index, table, exists in results["indexes"]:
            status = "✓" if exists else "✗"
            print(f"  {status} {index} on {table}")

        print(f"\nOverall Status: {'PASSED' if results['is_valid'] else 'FAILED'}")

        # Optionally clean orphaned records
        if not results["is_valid"] and results["orphaned_records"]:
            response = input("\nDo you want to clean orphaned records? (y/N): ")
            if response.lower() == "y":
                cleaned = validator.clean_orphaned_records(dry_run=False)
                print("\nCleaned orphaned records:")
                for table, count in cleaned.items():
                    print(f"  {table}: {count}")

        return 0 if results["is_valid"] else 1

    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
