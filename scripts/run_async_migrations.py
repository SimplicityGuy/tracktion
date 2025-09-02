#!/usr/bin/env python3
"""Script to run database migrations with async support."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Third-party imports
from alembic import command
from alembic.config import Config

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_database_url(use_async: bool = True) -> str:
    """Get database URL from environment.

    Args:
        use_async: Whether to return async URL

    Returns:
        Database URL
    """
    url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")

    if not url:
        url = "postgresql://user:password@localhost/tracktion"

    if use_async:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)

    return url


async def test_async_connection() -> bool:
    """Test async database connection.

    Returns:
        True if connection successful
    """
    try:
        engine = create_async_engine(get_database_url(use_async=True))
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        await engine.dispose()
        logger.info("✅ Async database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Async database connection failed: {e}")
        return False


def run_migrations(direction: str = "upgrade", revision: str = "head", use_async: bool = True) -> None:
    """Run database migrations.

    Args:
        direction: Migration direction (upgrade/downgrade)
        revision: Target revision
        use_async: Whether to use async migrations
    """
    # Setup Alembic config
    alembic_ini = project_root / "alembic.ini"
    alembic_cfg = Config(str(alembic_ini))

    # Set database URL
    alembic_cfg.set_main_option("sqlalchemy.url", get_database_url(use_async=False))

    # Use async env.py if requested
    if use_async:
        alembic_cfg.set_main_option("script_location", str(project_root / "alembic"))
        # Temporarily replace env.py with async_env.py
        env_py = project_root / "alembic" / "env.py"
        async_env_py = project_root / "alembic" / "async_env.py"

        if async_env_py.exists():
            # Backup original env.py
            env_backup = project_root / "alembic" / "env.py.backup"
            if env_py.exists():
                env_py.rename(env_backup)

            # Use async env
            async_env_py.rename(env_py)

            try:
                # Run migration
                if direction == "upgrade":
                    logger.info(f"Running async upgrade to {revision}")
                    command.upgrade(alembic_cfg, revision)
                elif direction == "downgrade":
                    logger.info(f"Running async downgrade to {revision}")
                    command.downgrade(alembic_cfg, revision)
                else:
                    raise ValueError(f"Invalid direction: {direction}")

                logger.info("✅ Migration completed successfully")

            finally:
                # Restore original env.py
                if env_backup.exists():
                    env_py.rename(async_env_py)
                    env_backup.rename(env_py)
        else:
            logger.warning("async_env.py not found, using standard migrations")
            if direction == "upgrade":
                command.upgrade(alembic_cfg, revision)
            elif direction == "downgrade":
                command.downgrade(alembic_cfg, revision)
    else:
        # Run standard migrations
        if direction == "upgrade":
            logger.info(f"Running sync upgrade to {revision}")
            command.upgrade(alembic_cfg, revision)
        elif direction == "downgrade":
            logger.info(f"Running sync downgrade to {revision}")
            command.downgrade(alembic_cfg, revision)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        logger.info("✅ Migration completed successfully")


def rollback_test() -> None:
    """Test rollback procedures."""
    logger.info("Testing rollback procedures...")

    try:
        # Get current revision
        alembic_ini = project_root / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option("sqlalchemy.url", get_database_url(use_async=False))

        # Show current revision
        logger.info("Current revision:")
        command.current(alembic_cfg)

        # Test downgrade
        logger.info("Testing downgrade...")
        command.downgrade(alembic_cfg, "-1")

        # Show new revision
        logger.info("After downgrade:")
        command.current(alembic_cfg)

        # Test upgrade
        logger.info("Testing upgrade...")
        command.upgrade(alembic_cfg, "head")

        # Show final revision
        logger.info("After upgrade:")
        command.current(alembic_cfg)

        logger.info("✅ Rollback test completed successfully")

    except Exception as e:
        logger.error(f"❌ Rollback test failed: {e}")
        raise


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run async database migrations")
    parser.add_argument(
        "command",
        choices=["upgrade", "downgrade", "test", "rollback-test"],
        help="Command to run",
    )
    parser.add_argument("--revision", default="head", help="Target revision (default: head)")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous migrations instead of async",
    )

    args = parser.parse_args()

    # Test connection first
    if not args.sync and not await test_async_connection():
        logger.error("Failed to connect to database with async driver")
        sys.exit(1)

    if args.command == "test":
        # Just test connection
        logger.info("Connection test completed")
    elif args.command == "rollback-test":
        rollback_test()
    else:
        run_migrations(direction=args.command, revision=args.revision, use_async=not args.sync)


if __name__ == "__main__":
    asyncio.run(main())
