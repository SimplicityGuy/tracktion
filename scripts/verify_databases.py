#!/usr/bin/env python
"""Script to verify database accessibility."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()


def verify_postgresql():
    """Verify PostgreSQL connectivity and schema."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set")
        return False
    
    print(f"Testing PostgreSQL connection...")
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Test basic connectivity
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            
            # Check if tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            
            expected_tables = ['alembic_version', 'metadata', 'recordings', 'tracklists']
            
            print(f"  ✅ Connected successfully")
            print(f"  📊 Tables found: {', '.join(tables)}")
            
            for table in expected_tables:
                if table in tables:
                    print(f"    ✅ {table}")
                else:
                    print(f"    ❌ {table} missing")
            
            return all(table in tables for table in expected_tables)
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def verify_neo4j():
    """Verify Neo4j connectivity."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "changeme")
    
    print(f"\nTesting Neo4j connection...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        
        # Check constraints
        with driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = list(result)
            
            print(f"  ✅ Connected successfully")
            print(f"  📊 Constraints found: {len(constraints)}")
            
            for constraint in constraints:
                print(f"    • {constraint['name']}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 50)
    print("DATABASE CONNECTIVITY VERIFICATION")
    print("=" * 50)
    
    pg_ok = verify_postgresql()
    neo4j_ok = verify_neo4j()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if pg_ok and neo4j_ok:
        print("✅ All databases are accessible and configured correctly!")
        return 0
    else:
        if not pg_ok:
            print("❌ PostgreSQL has issues")
        if not neo4j_ok:
            print("❌ Neo4j has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())