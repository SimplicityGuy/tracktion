#!/usr/bin/env python
"""Script to set up Neo4j constraints and indexes."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from shared.core_types.src.neo4j_repository import Neo4jRepository

# Load environment variables
load_dotenv()


def setup_neo4j():
    """Set up Neo4j constraints and indexes."""
    # Get connection details from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "changeme")

    print(f"Connecting to Neo4j at {uri}...")

    # Create repository and set up constraints
    repo = Neo4jRepository(uri, user, password)

    try:
        print("Creating constraints and indexes...")
        repo.create_constraints()
        print("✅ Neo4j setup complete!")
    except Exception as e:
        print(f"❌ Error setting up Neo4j: {e}")
        return 1
    finally:
        repo.close()

    return 0


if __name__ == "__main__":
    sys.exit(setup_neo4j())
