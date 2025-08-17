"""Neo4j repository for graph database operations."""

import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class Neo4jRepository:
    """Repository for Neo4j graph database operations."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        """Initialize Neo4j repository with connection details.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connectivity()

    def _verify_connectivity(self) -> None:
        """Verify Neo4j database connectivity.

        Raises:
            ServiceUnavailable: If Neo4j is not reachable
            AuthError: If authentication fails
        """
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully")
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close the driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_recording_node(self, recording_id: UUID, file_name: str, file_path: str) -> Dict[str, Any]:
        """Create a Recording node in Neo4j.

        Args:
            recording_id: UUID of the recording
            file_name: Name of the file
            file_path: Full path to the file

        Returns:
            Created node properties
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (r:Recording {
                    uuid: $uuid,
                    file_name: $file_name,
                    file_path: $file_path
                })
                RETURN r
                """,
                uuid=str(recording_id),
                file_name=file_name,
                file_path=file_path,
            )
            record = result.single()
            if record:
                return dict(record["r"])
            return {}

    def get_recording_node(self, recording_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a Recording node by UUID.

        Args:
            recording_id: UUID of the recording

        Returns:
            Node properties or None if not found
        """
        with self.driver.session() as session:
            result = session.run("MATCH (r:Recording {uuid: $uuid}) RETURN r", uuid=str(recording_id))
            record = result.single()
            if record:
                return dict(record["r"])
            return None

    def add_metadata_relationship(self, recording_id: UUID, key: str, value: str) -> None:
        """Create HAS_METADATA relationship with Metadata node.

        Args:
            recording_id: UUID of the recording
            key: Metadata key
            value: Metadata value
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (r:Recording {uuid: $uuid})
                CREATE (r)-[:HAS_METADATA]->(m:Metadata {key: $key, value: $value})
                """,
                uuid=str(recording_id),
                key=key,
                value=value,
            )
            logger.debug(f"Added metadata {key}={value} to recording {recording_id}")

    def get_recording_metadata(self, recording_id: UUID) -> List[Dict[str, str]]:
        """Get all metadata for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            List of metadata key-value pairs
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recording {uuid: $uuid})-[:HAS_METADATA]->(m:Metadata)
                RETURN m.key as key, m.value as value
                """,
                uuid=str(recording_id),
            )
            return [{"key": record["key"], "value": record["value"]} for record in result]

    def add_tracklist_with_tracks(self, recording_id: UUID, source: str, tracks: List[Dict[str, Any]]) -> None:
        """Create tracklist and track nodes with relationships.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist
            tracks: List of track dictionaries with title, artist, start_time
        """
        with self.driver.session() as session:
            # Create tracklist and tracks in a single transaction
            session.run(
                """
                MATCH (r:Recording {uuid: $recording_uuid})
                CREATE (r)-[:HAS_TRACKLIST]->(tl:Tracklist {source: $source})
                WITH tl
                UNWIND $tracks AS track
                CREATE (tl)-[:CONTAINS_TRACK {start_time: track.start_time}]->
                       (t:Track {title: track.title, artist: track.artist})
                """,
                recording_uuid=str(recording_id),
                source=source,
                tracks=tracks,
            )
            logger.debug(f"Added tracklist with {len(tracks)} tracks to recording {recording_id}")

    def get_tracklist_tracks(self, recording_id: UUID) -> List[Dict[str, Any]]:
        """Get all tracks for a recording's tracklist.

        Args:
            recording_id: UUID of the recording

        Returns:
            List of track information including start times
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recording {uuid: $uuid})-[:HAS_TRACKLIST]->(tl:Tracklist)
                      -[ct:CONTAINS_TRACK]->(t:Track)
                RETURN t.title as title, t.artist as artist,
                       ct.start_time as start_time
                ORDER BY ct.start_time
                """,
                uuid=str(recording_id),
            )
            return [
                {"title": record["title"], "artist": record["artist"], "start_time": record["start_time"]}
                for record in result
            ]

    def delete_recording_node(self, recording_id: UUID) -> bool:
        """Delete a recording node and all its relationships.

        Args:
            recording_id: UUID of the recording

        Returns:
            True if deleted, False if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Recording {uuid: $uuid})
                OPTIONAL MATCH (r)-[rel]->(n)
                DETACH DELETE r, rel, n
                RETURN count(r) as deleted
                """,
                uuid=str(recording_id),
            )
            record = result.single()
            return record["deleted"] > 0 if record else False

    def create_constraints(self) -> None:
        """Create database constraints and indexes for optimal performance."""
        with self.driver.session() as session:
            # Create uniqueness constraint on Recording UUID
            try:
                session.run(
                    """
                    CREATE CONSTRAINT recording_uuid_unique IF NOT EXISTS
                    FOR (r:Recording) REQUIRE r.uuid IS UNIQUE
                    """
                )
                logger.info("Created uniqueness constraint on Recording.uuid")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")

            # Create index on Metadata key for faster lookups
            try:
                session.run(
                    """
                    CREATE INDEX metadata_key_index IF NOT EXISTS
                    FOR (m:Metadata) ON (m.key)
                    """
                )
                logger.info("Created index on Metadata.key")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")

    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database.

        WARNING: This will delete all data in the database!
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared all data from Neo4j database")

    def recording_exists(self, recording_id: UUID) -> bool:
        """Check if a recording node exists.

        Args:
            recording_id: UUID of the recording

        Returns:
            True if the recording exists, False otherwise
        """
        with self.driver.session() as session:
            result = session.run("MATCH (r:Recording {uuid: $uuid}) RETURN count(r) as count", uuid=str(recording_id))
            record = result.single()
            return record["count"] > 0 if record else False

    def create_recording(
        self, recording_id: UUID, file_path: str, file_hash: str, properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a Recording node in Neo4j.

        Args:
            recording_id: UUID of the recording
            file_path: Full path to the file
            file_hash: SHA256 hash of the file
            properties: Additional properties for the node

        Returns:
            Created node properties
        """
        props = {"uuid": str(recording_id), "file_path": file_path, "file_hash": file_hash}
        if properties:
            props.update(properties)

        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (r:Recording $props)
                RETURN r
                """,
                props=props,
            )
            record = result.single()
            if record:
                return dict(record["r"])
            return {}

    def create_metadata(self, key: str, value: str, properties: Optional[Dict[str, Any]] = None) -> UUID:
        """Create a Metadata node.

        Args:
            key: Metadata key
            value: Metadata value
            properties: Additional properties for the node

        Returns:
            UUID of the created metadata node
        """
        import uuid

        metadata_id = uuid.uuid4()
        props = {"uuid": str(metadata_id), "key": key, "value": value}
        if properties:
            props.update(properties)

        with self.driver.session() as session:
            session.run(
                """
                CREATE (m:Metadata $props)
                """,
                props=props,
            )
        return metadata_id

    def create_has_metadata_relationship(
        self, recording_id: UUID, metadata_id: UUID, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create HAS_METADATA relationship between Recording and Metadata.

        Args:
            recording_id: UUID of the recording
            metadata_id: UUID of the metadata
            properties: Additional properties for the relationship
        """
        with self.driver.session() as session:
            if properties:
                session.run(
                    """
                    MATCH (r:Recording {uuid: $recording_uuid})
                    MATCH (m:Metadata {uuid: $metadata_uuid})
                    CREATE (r)-[:HAS_METADATA $props]->(m)
                    """,
                    recording_uuid=str(recording_id),
                    metadata_uuid=str(metadata_id),
                    props=properties,
                )
            else:
                session.run(
                    """
                    MATCH (r:Recording {uuid: $recording_uuid})
                    MATCH (m:Metadata {uuid: $metadata_uuid})
                    CREATE (r)-[:HAS_METADATA]->(m)
                    """,
                    recording_uuid=str(recording_id),
                    metadata_uuid=str(metadata_id),
                )

    def create_or_get_artist(self, name: str) -> UUID:
        """Create or get an Artist node.

        Args:
            name: Artist name

        Returns:
            UUID of the artist node
        """
        import uuid

        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (a:Artist {name: $name})
                ON CREATE SET a.uuid = $uuid
                RETURN a.uuid as uuid
                """,
                name=name,
                uuid=str(uuid.uuid4()),
            )
            record = result.single()
            if record and record["uuid"]:
                return UUID(record["uuid"])
            return uuid.uuid4()

    def create_or_get_album(self, name: str, artist: Optional[str] = None) -> UUID:
        """Create or get an Album node.

        Args:
            name: Album name
            artist: Optional artist name

        Returns:
            UUID of the album node
        """
        import uuid

        with self.driver.session() as session:
            if artist:
                result = session.run(
                    """
                    MERGE (a:Album {name: $name, artist: $artist})
                    ON CREATE SET a.uuid = $uuid
                    RETURN a.uuid as uuid
                    """,
                    name=name,
                    artist=artist,
                    uuid=str(uuid.uuid4()),
                )
            else:
                result = session.run(
                    """
                    MERGE (a:Album {name: $name})
                    ON CREATE SET a.uuid = $uuid
                    RETURN a.uuid as uuid
                    """,
                    name=name,
                    uuid=str(uuid.uuid4()),
                )
            record = result.single()
            if record and record["uuid"]:
                return UUID(record["uuid"])
            return uuid.uuid4()

    def create_or_get_genre(self, name: str) -> UUID:
        """Create or get a Genre node.

        Args:
            name: Genre name

        Returns:
            UUID of the genre node
        """
        import uuid

        with self.driver.session() as session:
            result = session.run(
                """
                MERGE (g:Genre {name: $name})
                ON CREATE SET g.uuid = $uuid
                RETURN g.uuid as uuid
                """,
                name=name,
                uuid=str(uuid.uuid4()),
            )
            record = result.single()
            if record and record["uuid"]:
                return UUID(record["uuid"])
            return uuid.uuid4()

    def create_relationship(
        self, from_id: UUID, to_id: UUID, relationship_type: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create a relationship between two nodes.

        Args:
            from_id: UUID of the source node
            to_id: UUID of the target node
            relationship_type: Type of the relationship
            properties: Optional properties for the relationship
        """
        with self.driver.session() as session:
            if properties:
                session.run(
                    f"""
                    MATCH (from {{uuid: $from_uuid}})
                    MATCH (to {{uuid: $to_uuid}})
                    CREATE (from)-[:{relationship_type} $props]->(to)
                    """,
                    from_uuid=str(from_id),
                    to_uuid=str(to_id),
                    props=properties,
                )
            else:
                session.run(
                    f"""
                    MATCH (from {{uuid: $from_uuid}})
                    MATCH (to {{uuid: $to_uuid}})
                    CREATE (from)-[:{relationship_type}]->(to)
                    """,
                    from_uuid=str(from_id),
                    to_uuid=str(to_id),
                )

    def update_recording_properties(self, recording_id: UUID, properties: Dict[str, Any]) -> None:
        """Update properties of a Recording node.

        Args:
            recording_id: UUID of the recording
            properties: Properties to update
        """
        with self.driver.session() as session:
            session.run(
                """
                MATCH (r:Recording {uuid: $uuid})
                SET r += $props
                """,
                uuid=str(recording_id),
                props=properties,
            )
