"""Async storage handler for analysis service metadata."""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from neo4j import AsyncGraphDatabase
from redis import asyncio as aioredis
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.async_repositories import (
    AsyncRecordingRepository,
    AsyncMetadataRepository,
    AsyncTracklistRepository,
)

logger = logging.getLogger(__name__)


class AsyncNeo4jRepository:
    """Async repository for Neo4j graph database operations."""

    def __init__(self, uri: str, auth: tuple, database: str = "neo4j") -> None:
        """Initialize async Neo4j repository.

        Args:
            uri: Neo4j connection URI
            auth: Authentication tuple (username, password)
            database: Database name
        """
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self.database = database

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        await self.driver.close()

    async def create_recording_node(self, recording_id: UUID, file_path: str, file_name: str) -> Dict[str, Any]:
        """Create a Recording node in Neo4j.

        Args:
            recording_id: UUID of the recording
            file_path: Path to the file
            file_name: Name of the file

        Returns:
            Created node properties
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MERGE (r:Recording {uuid: $uuid})
            SET r.file_path = $file_path,
                r.file_name = $file_name,
                r.created_at = datetime()
            RETURN r
            """
            result = await session.run(query, uuid=str(recording_id), file_path=file_path, file_name=file_name)
            record = await result.single()
            return dict(record["r"]) if record else {}

    async def create_metadata_relationships(self, recording_id: UUID, metadata: List[Dict[str, str]]) -> int:
        """Create metadata nodes and relationships in Neo4j.

        Args:
            recording_id: UUID of the recording
            metadata: List of metadata key-value pairs

        Returns:
            Number of relationships created
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Recording {uuid: $uuid})
            UNWIND $metadata AS meta
            MERGE (m:Metadata {key: meta.key, value: meta.value})
            MERGE (r)-[:HAS_METADATA]->(m)
            RETURN count(*) as count
            """
            result = await session.run(query, uuid=str(recording_id), metadata=metadata)
            record = await result.single()
            return record["count"] if record else 0

    async def create_tracklist_relationship(
        self, recording_id: UUID, source: str, tracks: List[Dict[str, Any]]
    ) -> bool:
        """Create tracklist node and relationship in Neo4j.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist
            tracks: List of track information

        Returns:
            True if created successfully
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r:Recording {uuid: $uuid})
            MERGE (t:Tracklist {source: $source})
            SET t.track_count = $track_count,
                t.updated_at = datetime()
            MERGE (r)-[:HAS_TRACKLIST]->(t)
            WITH r, t
            UNWIND $tracks AS track
            MERGE (tr:Track {name: track.name, artist: track.artist})
            MERGE (t)-[:CONTAINS_TRACK {position: track.position}]->(tr)
            RETURN count(*) as count
            """
            result = await session.run(
                query, uuid=str(recording_id), source=source, track_count=len(tracks), tracks=tracks
            )
            record = await result.single()
            return bool(record and record["count"] > 0)

    async def find_similar_recordings(self, recording_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """Find recordings with similar metadata.

        Args:
            recording_id: UUID of the recording
            limit: Maximum number of results

        Returns:
            List of similar recordings
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH (r1:Recording {uuid: $uuid})-[:HAS_METADATA]->(m:Metadata)
            <-[:HAS_METADATA]-(r2:Recording)
            WHERE r1 <> r2
            WITH r2, count(m) as shared_metadata
            ORDER BY shared_metadata DESC
            LIMIT $limit
            RETURN r2.uuid as uuid, r2.file_name as file_name,
                   shared_metadata
            """
            result = await session.run(query, uuid=str(recording_id), limit=limit)
            return [dict(record) async for record in result]

    async def get_recording_graph(self, recording_id: UUID, depth: int = 2) -> Dict[str, Any]:
        """Get the graph structure around a recording.

        Args:
            recording_id: UUID of the recording
            depth: Depth of relationships to traverse

        Returns:
            Graph structure as dictionary
        """
        async with self.driver.session(database=self.database) as session:
            query = """
            MATCH path = (r:Recording {uuid: $uuid})-[*1..$depth]-(n)
            RETURN r, relationships(path) as rels, nodes(path) as nodes
            """
            result = await session.run(query, uuid=str(recording_id), depth=depth)

            graph = {"recording": {}, "relationships": [], "nodes": []}

            async for record in result:
                if not graph["recording"]:
                    graph["recording"] = dict(record["r"])

                for rel in record["rels"]:
                    rel_dict = {
                        "type": rel.type,
                        "start": rel.start_node.id,
                        "end": rel.end_node.id,
                        "properties": dict(rel),
                    }
                    if rel_dict not in graph["relationships"]:
                        graph["relationships"].append(rel_dict)

                for node in record["nodes"]:
                    node_dict = {"id": node.id, "labels": list(node.labels), "properties": dict(node)}
                    if node_dict not in graph["nodes"]:
                        graph["nodes"].append(node_dict)

            return graph


class AsyncRedisCache:
    """Async Redis cache for analysis results."""

    def __init__(self, redis_url: str) -> None:
        """Initialize async Redis cache.

        Args:
            redis_url: Redis connection URL
        """
        self.redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True, max_connections=50)

    async def close(self) -> None:
        """Close Redis connection."""
        await self.redis.close()

    async def set_analysis_result(self, key: str, result: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache analysis result.

        Args:
            key: Cache key
            result: Analysis result to cache
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        try:
            import json

            return await self.redis.setex(f"analysis:{key}", ttl, json.dumps(result))
        except Exception as e:
            logger.error(f"Error caching analysis result: {e}")
            return False

    async def get_analysis_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        try:
            import json

            result = await self.redis.get(f"analysis:{key}")
            return json.loads(result) if result else None
        except Exception as e:
            logger.error(f"Error retrieving cached result: {e}")
            return None

    async def invalidate_recording_cache(self, recording_id: UUID) -> int:
        """Invalidate all cache entries for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Number of keys deleted
        """
        pattern = f"analysis:*{recording_id}*"
        keys = []
        async for key in self.redis.scan_iter(match=pattern):
            keys.append(key)

        if keys:
            return await self.redis.delete(*keys)
        return 0


class AsyncStorageHandler:
    """Async handler for storing analysis metadata."""

    def __init__(self, postgres_url: str, neo4j_uri: str, neo4j_auth: tuple, redis_url: str) -> None:
        """Initialize async storage handler.

        Args:
            postgres_url: PostgreSQL connection URL
            neo4j_uri: Neo4j connection URI
            neo4j_auth: Neo4j authentication tuple
            redis_url: Redis connection URL
        """
        # PostgreSQL repositories
        self.db_manager = AsyncDatabaseManager(database_url=postgres_url)
        self.recording_repo = AsyncRecordingRepository(self.db_manager)
        self.metadata_repo = AsyncMetadataRepository(self.db_manager)
        self.tracklist_repo = AsyncTracklistRepository(self.db_manager)

        # Neo4j repository
        self.neo4j_repo = AsyncNeo4jRepository(neo4j_uri, neo4j_auth)

        # Redis cache
        self.redis_cache = AsyncRedisCache(redis_url)

    async def close(self) -> None:
        """Close all connections."""
        await self.neo4j_repo.close()
        await self.redis_cache.close()

    async def store_analysis_results(self, recording_id: UUID, analysis_type: str, results: Dict[str, Any]) -> bool:
        """Store analysis results across databases.

        Args:
            recording_id: UUID of the recording
            analysis_type: Type of analysis performed
            results: Analysis results

        Returns:
            True if stored successfully
        """
        try:
            # Get recording from PostgreSQL
            recording = await self.recording_repo.get_by_id(recording_id)
            if not recording:
                logger.error(f"Recording {recording_id} not found")
                return False

            # Store metadata in PostgreSQL
            if "metadata" in results:
                for key, value in results["metadata"].items():
                    await self.metadata_repo.create(
                        recording_id=recording_id, key=f"{analysis_type}_{key}", value=str(value)
                    )

            # Create graph relationships in Neo4j
            await self.neo4j_repo.create_recording_node(
                recording_id=recording_id, file_path=recording.file_path, file_name=recording.file_name
            )

            if "metadata" in results:
                metadata_list = [{"key": k, "value": str(v)} for k, v in results["metadata"].items()]
                await self.neo4j_repo.create_metadata_relationships(recording_id=recording_id, metadata=metadata_list)

            # Cache results in Redis
            cache_key = f"{recording_id}:{analysis_type}"
            await self.redis_cache.set_analysis_result(key=cache_key, result=results, ttl=3600)

            logger.info(f"Stored analysis results for {recording_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
            return False

    async def get_cached_analysis(self, recording_id: UUID, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results.

        Args:
            recording_id: UUID of the recording
            analysis_type: Type of analysis

        Returns:
            Cached results or None
        """
        cache_key = f"{recording_id}:{analysis_type}"
        return await self.redis_cache.get_analysis_result(cache_key)

    async def find_similar_recordings(self, recording_id: UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """Find recordings with similar characteristics.

        Args:
            recording_id: UUID of the recording
            limit: Maximum number of results

        Returns:
            List of similar recordings
        """
        # Try cache first
        cache_key = f"{recording_id}:similar"
        cached = await self.redis_cache.get_analysis_result(cache_key)
        if cached:
            return cached.get("recordings", [])

        # Query Neo4j for similar recordings
        similar = await self.neo4j_repo.find_similar_recordings(recording_id=recording_id, limit=limit)

        # Cache the results
        await self.redis_cache.set_analysis_result(
            key=cache_key,
            result={"recordings": similar},
            ttl=1800,  # 30 minutes
        )

        return similar

    async def store_tracklist(self, recording_id: UUID, source: str, tracks: List[Dict[str, Any]]) -> bool:
        """Store tracklist information.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist
            tracks: List of track information

        Returns:
            True if stored successfully
        """
        try:
            # Store in PostgreSQL
            await self.tracklist_repo.create(recording_id=recording_id, source=source, tracks={"tracks": tracks})

            # Create graph relationships in Neo4j
            await self.neo4j_repo.create_tracklist_relationship(recording_id=recording_id, source=source, tracks=tracks)

            # Invalidate related cache
            await self.redis_cache.invalidate_recording_cache(recording_id)

            logger.info(f"Stored tracklist for {recording_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing tracklist: {e}")
            return False

    async def get_recording_graph(self, recording_id: UUID, depth: int = 2) -> Dict[str, Any]:
        """Get the graph structure around a recording.

        Args:
            recording_id: UUID of the recording
            depth: Depth of relationships to traverse

        Returns:
            Graph structure
        """
        # Try cache first
        cache_key = f"{recording_id}:graph:{depth}"
        cached = await self.redis_cache.get_analysis_result(cache_key)
        if cached:
            return cached

        # Query Neo4j
        graph = await self.neo4j_repo.get_recording_graph(recording_id=recording_id, depth=depth)

        # Cache the results
        await self.redis_cache.set_analysis_result(
            key=cache_key,
            result=graph,
            ttl=600,  # 10 minutes
        )

        return graph
