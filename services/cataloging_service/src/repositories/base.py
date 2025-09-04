"""Base repository class with common CRUD operations."""

from typing import Any, Generic, TypeVar
from uuid import UUID

from services.cataloging_service.src.models.base import Base
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    def __init__(self, model: type[ModelType], session: AsyncSession) -> None:
        """Initialize the base repository.

        Args:
            model: The SQLAlchemy model class
            session: The database session
        """
        self.model = model
        self.session = session

    async def create(self, **kwargs: Any) -> ModelType:
        """Create a new record.

        Args:
            **kwargs: Field values for the new record

        Returns:
            The created record
        """
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get_by_id(self, id: UUID) -> ModelType | None:
        """Get a record by ID.

        Args:
            id: The record ID

        Returns:
            The record if found, None otherwise
        """
        result = await self.session.execute(select(self.model).where(self.model.id == id))  # All models have id field
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[ModelType]:
        """Get all records with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of records
        """
        result = await self.session.execute(select(self.model).limit(limit).offset(offset))
        return list(result.scalars().all())

    async def update(self, id: UUID, **kwargs: Any) -> ModelType | None:
        """Update a record by ID.

        Args:
            id: The record ID
            **kwargs: Field values to update

        Returns:
            The updated record if found, None otherwise
        """
        instance = await self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            await self.session.flush()
        return instance

    async def delete(self, id: UUID) -> bool:
        """Delete a record by ID.

        Args:
            id: The record ID

        Returns:
            True if the record was deleted, False otherwise
        """
        instance = await self.get_by_id(id)
        if instance:
            await self.session.delete(instance)
            await self.session.flush()
            return True
        return False

    async def count(self) -> int:
        """Count total records.

        Returns:
            Total number of records
        """
        result = await self.session.execute(select(func.count()).select_from(self.model))
        return result.scalar() or 0
