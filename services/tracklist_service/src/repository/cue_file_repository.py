"""
CUE file repository for database operations.
"""

import logging
from typing import cast
from uuid import UUID

from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from services.tracklist_service.src.models.cue_file import CueFileDB

logger = logging.getLogger(__name__)


class CueFileRepository:
    """Repository for CUE file database operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.

        Args:
            session: Async database session
        """
        self.session = session

    async def create_cue_file(self, cue_file_db: CueFileDB) -> CueFileDB:
        """
        Create a new CUE file record.

        Args:
            cue_file_db: CUE file database model

        Returns:
            Created CUE file record
        """
        try:
            # Check if file already exists and increment version
            existing_files = await self.get_cue_files_by_tracklist_and_format(
                cue_file_db.tracklist_id,
                cue_file_db.format,
            )

            if existing_files:
                # Mark previous versions as inactive
                for existing_file in existing_files:
                    existing_file.is_active = False  # type: ignore[assignment]  # SQLAlchemy instance attribute assignment at runtime

                # Set new version number
                latest_version = max(f.version for f in existing_files)
                cue_file_db.version = latest_version + 1  # type: ignore[assignment]  # SQLAlchemy instance attribute assignment at runtime
            else:
                cue_file_db.version = 1  # type: ignore[assignment]  # SQLAlchemy instance attribute assignment at runtime

            # Ensure new file is active
            cue_file_db.is_active = True  # type: ignore[assignment]  # SQLAlchemy instance attribute assignment at runtime

            self.session.add(cue_file_db)
            await self.session.commit()
            await self.session.refresh(cue_file_db)

            logger.info(f"Created CUE file {cue_file_db.id} version {cue_file_db.version}")
            return cue_file_db

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create CUE file: {e}", exc_info=True)
            raise

    async def get_cue_file_by_id(self, cue_file_id: UUID) -> CueFileDB | None:
        """
        Get CUE file by ID.

        Args:
            cue_file_id: CUE file ID

        Returns:
            CUE file record or None if not found
        """
        try:
            result = await self.session.execute(select(CueFileDB).where(CueFileDB.id == cue_file_id))
            cue_file = result.scalar_one_or_none()

            if cue_file:
                logger.debug(f"Found CUE file {cue_file_id}")
            else:
                logger.debug(f"CUE file {cue_file_id} not found")

            return cast("CueFileDB | None", cue_file)

        except Exception as e:
            logger.error(f"Failed to get CUE file {cue_file_id}: {e}", exc_info=True)
            raise

    async def get_cue_files_by_tracklist(self, tracklist_id: UUID) -> list[CueFileDB]:
        """
        Get all CUE files for a tracklist.

        Args:
            tracklist_id: Tracklist ID

        Returns:
            List of CUE files
        """
        try:
            result = await self.session.execute(
                select(CueFileDB)
                .where(CueFileDB.tracklist_id == tracklist_id)
                .order_by(CueFileDB.format, desc(CueFileDB.version))
            )
            cue_files = result.scalars().all()

            logger.debug(f"Found {len(cue_files)} CUE files for tracklist {tracklist_id}")
            return list(cue_files)

        except Exception as e:
            logger.error(
                f"Failed to get CUE files for tracklist {tracklist_id}: {e}",
                exc_info=True,
            )
            raise

    async def get_cue_files_by_tracklist_and_format(self, tracklist_id: UUID, cue_format: str) -> list[CueFileDB]:
        """
        Get CUE files by tracklist and format.

        Args:
            tracklist_id: Tracklist ID
            cue_format: CUE format

        Returns:
            List of CUE files
        """
        try:
            result = await self.session.execute(
                select(CueFileDB)
                .where(
                    and_(
                        CueFileDB.tracklist_id == tracklist_id,
                        CueFileDB.format == cue_format,
                    )
                )
                .order_by(desc(CueFileDB.version))
            )
            cue_files = result.scalars().all()

            logger.debug(f"Found {len(cue_files)} CUE files for tracklist {tracklist_id} format {cue_format}")
            return list(cue_files)

        except Exception as e:
            logger.error(
                f"Failed to get CUE files for tracklist {tracklist_id} format {cue_format}: {e}",
                exc_info=True,
            )
            raise

    async def get_active_cue_file(self, tracklist_id: UUID, cue_format: str) -> CueFileDB | None:
        """
        Get active CUE file for tracklist and format.

        Args:
            tracklist_id: Tracklist ID
            cue_format: CUE format

        Returns:
            Active CUE file or None
        """
        try:
            result = await self.session.execute(
                select(CueFileDB).where(
                    and_(
                        CueFileDB.tracklist_id == tracklist_id,
                        CueFileDB.format == cue_format,
                        CueFileDB.is_active.is_(True),
                    )
                )
            )
            cue_file = result.scalar_one_or_none()

            if cue_file:
                logger.debug(f"Found active CUE file {cue_file.id} for tracklist {tracklist_id}")
            else:
                logger.debug(f"No active CUE file found for tracklist {tracklist_id} format {cue_format}")

            return cast("CueFileDB | None", cue_file)

        except Exception as e:
            logger.error(
                f"Failed to get active CUE file for tracklist {tracklist_id} format {cue_format}: {e}",
                exc_info=True,
            )
            raise

    async def update_cue_file(self, cue_file_db: CueFileDB) -> CueFileDB:
        """
        Update CUE file record.

        Args:
            cue_file_db: Updated CUE file database model

        Returns:
            Updated CUE file record
        """
        try:
            await self.session.commit()
            await self.session.refresh(cue_file_db)

            logger.info(f"Updated CUE file {cue_file_db.id}")
            return cue_file_db

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update CUE file {cue_file_db.id}: {e}", exc_info=True)
            raise

    async def soft_delete_cue_file(self, cue_file_id: UUID) -> bool:
        """
        Soft delete CUE file by marking as inactive.

        Args:
            cue_file_id: CUE file ID

        Returns:
            True if deleted successfully
        """
        try:
            result = await self.session.execute(select(CueFileDB).where(CueFileDB.id == cue_file_id))
            cue_file = result.scalar_one_or_none()

            if not cue_file:
                logger.warning(f"CUE file {cue_file_id} not found for soft delete")
                return False

            cue_file.is_active = False
            await self.session.commit()

            logger.info(f"Soft deleted CUE file {cue_file_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to soft delete CUE file {cue_file_id}: {e}", exc_info=True)
            raise

    async def hard_delete_cue_file(self, cue_file_id: UUID) -> bool:
        """
        Hard delete CUE file from database.

        Args:
            cue_file_id: CUE file ID

        Returns:
            True if deleted successfully
        """
        try:
            result = await self.session.execute(select(CueFileDB).where(CueFileDB.id == cue_file_id))
            cue_file = result.scalar_one_or_none()

            if not cue_file:
                logger.warning(f"CUE file {cue_file_id} not found for hard delete")
                return False

            await self.session.delete(cue_file)
            await self.session.commit()

            logger.info(f"Hard deleted CUE file {cue_file_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to hard delete CUE file {cue_file_id}: {e}", exc_info=True)
            raise

    async def list_cue_files(
        self,
        tracklist_id: UUID | None = None,
        cue_format: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[CueFileDB]:
        """
        List CUE files with filtering and pagination.

        Args:
            tracklist_id: Filter by tracklist ID
            cue_format: Filter by format
            limit: Maximum results
            offset: Result offset

        Returns:
            List of CUE files
        """
        try:
            query = select(CueFileDB)

            # Apply filters
            if tracklist_id:
                query = query.where(CueFileDB.tracklist_id == tracklist_id)

            if cue_format:
                query = query.where(CueFileDB.format == cue_format)

            # Apply pagination
            query = query.order_by(desc(CueFileDB.created_at)).limit(limit).offset(offset)

            result = await self.session.execute(query)
            cue_files = result.scalars().all()

            logger.debug(f"Listed {len(cue_files)} CUE files with filters")
            return list(cue_files)

        except Exception as e:
            logger.error(f"Failed to list CUE files: {e}", exc_info=True)
            raise

    async def count_cue_files(self, tracklist_id: UUID | None = None, cue_format: str | None = None) -> int:
        """
        Count CUE files with filtering.

        Args:
            tracklist_id: Filter by tracklist ID
            cue_format: Filter by format

        Returns:
            Total count of matching CUE files
        """
        try:
            query = select(func.count(CueFileDB.id))

            # Apply filters
            if tracklist_id:
                query = query.where(CueFileDB.tracklist_id == tracklist_id)

            if cue_format:
                query = query.where(CueFileDB.format == cue_format)

            result = await self.session.execute(query)
            count = result.scalar()

            logger.debug(f"Counted {count} CUE files with filters")
            return count or 0

        except Exception as e:
            logger.error(f"Failed to count CUE files: {e}", exc_info=True)
            raise

    async def get_file_versions(self, cue_file_id: UUID) -> list[CueFileDB]:
        """
        Get all versions of a CUE file.

        Args:
            cue_file_id: CUE file ID

        Returns:
            List of file versions ordered by version number
        """
        try:
            # First get the base file to find tracklist and format
            base_file = await self.get_cue_file_by_id(cue_file_id)
            if not base_file:
                return []

            # Get all versions for this tracklist/format combination
            result = await self.session.execute(
                select(CueFileDB)
                .where(
                    and_(
                        CueFileDB.tracklist_id == base_file.tracklist_id,
                        CueFileDB.format == base_file.format,
                    )
                )
                .order_by(desc(CueFileDB.version))
            )
            versions = result.scalars().all()

            logger.debug(f"Found {len(versions)} versions for CUE file {cue_file_id}")
            return list(versions)

        except Exception as e:
            logger.error(f"Failed to get versions for CUE file {cue_file_id}: {e}", exc_info=True)
            raise
