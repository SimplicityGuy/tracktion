"""Template manager for user-defined naming templates."""

import logging
import re
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from .models import NamingTemplate

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages user-defined naming templates with validation and usage tracking."""

    def __init__(self) -> None:
        """Initialize the template manager with in-memory storage."""
        # TODO: Replace with PostgreSQL integration
        self._templates: dict[str, NamingTemplate] = {}
        self._user_templates: dict[str, list[str]] = defaultdict(list)  # user_id -> template_ids
        self._usage_stats: dict[str, int] = {}  # template_id -> usage_count

        # Cache for parsed template variables to avoid re-parsing
        self._variable_cache: dict[str, list[str]] = {}

        # Initialize with default templates
        self._initialize_default_templates()

    def _initialize_default_templates(self) -> None:
        """Initialize the manager with default naming templates."""
        default_templates = self.get_default_templates()
        for template in default_templates:
            self._templates[template.id] = template
            self._user_templates["system"].append(template.id)

    async def save_template(self, template: NamingTemplate) -> str:
        """Save a naming template.

        Args:
            template: The naming template to save

        Returns:
            The template ID

        Raises:
            ValueError: If the template pattern is invalid
        """
        # Validate the template pattern
        if not self.validate_template_pattern(template.pattern):
            raise ValueError(f"Invalid template pattern: {template.pattern}")

        # Generate ID if not provided
        if not template.id:
            template.id = str(uuid.uuid4())

        # Store template
        self._templates[template.id] = template
        self._user_templates[template.user_id].append(template.id)
        self._usage_stats[template.id] = template.usage_count

        # Cache parsed variables
        self._variable_cache[template.id] = self.parse_template_variables(template.pattern)

        logger.info(f"Saved template '{template.name}' (ID: {template.id}) for user {template.user_id}")

        # TODO: Persist to PostgreSQL
        # async with get_db_session() as session:
        #     db_template = TemplateModel(**template.model_dump())
        #     session.add(db_template)
        #     await session.commit()

        return template.id

    async def get_template(self, template_id: str) -> NamingTemplate | None:
        """Retrieve a template by ID.

        Args:
            template_id: The template identifier

        Returns:
            The naming template or None if not found
        """
        # TODO: Query from PostgreSQL
        # async with get_db_session() as session:
        #     result = await session.execute(
        #         select(TemplateModel).where(TemplateModel.id == template_id)
        #     )
        #     db_template = result.scalar_one_or_none()
        #     return NamingTemplate.model_validate(db_template) if db_template else None

        return self._templates.get(template_id)

    async def get_user_templates(self, user_id: str) -> list[NamingTemplate]:
        """Get all templates for a specific user.

        Args:
            user_id: The user identifier

        Returns:
            List of naming templates for the user
        """
        # TODO: Query from PostgreSQL with proper filtering and ordering
        # async with get_db_session() as session:
        #     result = await session.execute(
        #         select(TemplateModel)
        #         .where(TemplateModel.user_id == user_id)
        #         .where(TemplateModel.is_active == True)
        #         .order_by(TemplateModel.usage_count.desc(), TemplateModel.created_at.desc())
        #     )
        #     db_templates = result.scalars().all()
        #     return [NamingTemplate.model_validate(t) for t in db_templates]

        template_ids = self._user_templates.get(user_id, [])
        templates = [self._templates[tid] for tid in template_ids if tid in self._templates]

        # Sort by usage count (desc) then creation time (desc)
        return sorted([t for t in templates if t.is_active], key=lambda t: (t.usage_count, t.created_at), reverse=True)

    async def update_usage_count(self, template_id: str) -> None:
        """Increment the usage count for a template.

        Args:
            template_id: The template identifier
        """
        if template_id in self._templates:
            template = self._templates[template_id]
            template.usage_count += 1
            self._usage_stats[template_id] = template.usage_count

            logger.debug(f"Incremented usage count for template {template_id} to {template.usage_count}")

            # TODO: Update in PostgreSQL
            # async with get_db_session() as session:
            #     await session.execute(
            #         update(TemplateModel)
            #         .where(TemplateModel.id == template_id)
            #         .values(usage_count=TemplateModel.usage_count + 1)
            #     )
            #     await session.commit()

    async def delete_template(self, template_id: str, user_id: str) -> bool:
        """Delete a template (soft delete by setting is_active=False).

        Args:
            template_id: The template identifier
            user_id: The user identifier (for authorization)

        Returns:
            True if deleted, False if not found or not authorized
        """
        template = await self.get_template(template_id)
        if not template or template.user_id != user_id:
            return False

        # Soft delete
        template.is_active = False
        self._templates[template_id] = template

        # Remove from user's active templates list
        if template_id in self._user_templates[user_id]:
            self._user_templates[user_id].remove(template_id)

        logger.info(f"Soft deleted template {template_id} for user {user_id}")

        # TODO: Update in PostgreSQL
        # async with get_db_session() as session:
        #     await session.execute(
        #         update(TemplateModel)
        #         .where(TemplateModel.id == template_id)
        #         .where(TemplateModel.user_id == user_id)
        #         .values(is_active=False)
        #     )
        #     await session.commit()

        return True

    def validate_template_pattern(self, pattern: str) -> bool:
        """Validate a template pattern for correct syntax.

        Args:
            pattern: The template pattern string

        Returns:
            True if valid, False otherwise
        """
        if not pattern or not isinstance(pattern, str):
            return False

        try:
            # Check for balanced braces
            brace_count = 0
            in_variable = False
            variable_name = ""

            for _i, char in enumerate(pattern):
                if char == "{":
                    if in_variable:
                        return False  # Nested braces not allowed
                    brace_count += 1
                    in_variable = True
                    variable_name = ""
                elif char == "}":
                    if not in_variable:
                        return False  # Closing brace without opening
                    brace_count -= 1
                    in_variable = False

                    # Validate variable name
                    if not self._is_valid_variable_name(variable_name):
                        return False
                elif in_variable:
                    variable_name += char

            # Check if all braces are balanced
            if brace_count != 0:
                return False

            # Try to format with dummy values to catch any other issues
            variables = self.parse_template_variables(pattern)
            dummy_values = {var: f"test_{var}" for var in variables}
            pattern.format(**dummy_values)

            return True

        except (ValueError, KeyError, TypeError) as e:
            logger.debug(f"Template pattern validation failed: {e}")
            return False

    def _is_valid_variable_name(self, name: str) -> bool:
        """Check if a variable name is valid.

        Args:
            name: The variable name to validate

        Returns:
            True if valid, False otherwise
        """
        if not name:
            return False

        # Must be a valid Python identifier (alphanumeric + underscore, not starting with digit)
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return False

        # Check against known variable types (can be extended)
        known_variables = {
            "artist",
            "date",
            "venue",
            "quality",
            "format",
            "source",
            "track",
            "set",
            "tour",
            "label",
            "catalog",
            "title",
            "album",
            "year",
            "month",
            "day",
            "genre",
            "bitrate",
            "sample_rate",
            "channels",
            "duration",
            "file_size",
            "taper",
            "lineage",
            "generation",
            "equipment",
            "notes",
            "show_date",
            "setlist",
        }

        return name.lower() in known_variables

    def parse_template_variables(self, pattern: str) -> list[str]:
        """Parse and return all variables from a template pattern.

        Args:
            pattern: The template pattern string

        Returns:
            List of variable names found in the pattern
        """
        if not pattern:
            return []

        # Use regex to find all variables in braces
        variables = re.findall(r"\{([^}]+)\}", pattern)

        # Remove duplicates while preserving order
        seen = set()
        unique_variables = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_variables.append(var)

        return unique_variables

    def get_default_templates(self) -> list[NamingTemplate]:
        """Get the default naming templates.

        Returns:
            List of default naming templates
        """
        now = datetime.now(UTC)

        return [
            NamingTemplate(
                id="default_concert_basic",
                name="Concert Basic",
                pattern="{artist} - {date} - {venue}",
                user_id="system",
                created_at=now,
                description="Basic concert recording format with artist, date, and venue",
                usage_count=0,
            ),
            NamingTemplate(
                id="default_concert_detailed",
                name="Concert Detailed",
                pattern="{artist} - {date} - {venue} - {quality} - {source}",
                user_id="system",
                created_at=now,
                description="Detailed concert format including quality and source information",
                usage_count=0,
            ),
            NamingTemplate(
                id="default_album_studio",
                name="Studio Album",
                pattern="{artist} - {year} - {album} - {format}",
                user_id="system",
                created_at=now,
                description="Studio album format with artist, year, album name, and format",
                usage_count=0,
            ),
            NamingTemplate(
                id="default_bootleg_complete",
                name="Bootleg Complete",
                pattern="{artist} - {date} - {venue} - {set} - {quality} - {source} - {generation}",
                user_id="system",
                created_at=now,
                description="Complete bootleg format with all available metadata",
                usage_count=0,
            ),
            NamingTemplate(
                id="default_track_numbered",
                name="Track with Numbers",
                pattern="{track:02d} - {artist} - {title}",
                user_id="system",
                created_at=now,
                description="Individual track format with zero-padded track number",
                usage_count=0,
            ),
            NamingTemplate(
                id="default_date_sortable",
                name="Date Sortable",
                pattern="{year}-{month:02d}-{day:02d} - {artist} - {venue}",
                user_id="system",
                created_at=now,
                description="Date-first format for chronological sorting",
                usage_count=0,
            ),
        ]

    async def search_templates(self, user_id: str, query: str) -> list[NamingTemplate]:
        """Search templates by name or description.

        Args:
            user_id: The user identifier
            query: Search query string

        Returns:
            List of matching templates
        """
        if not query:
            return await self.get_user_templates(user_id)

        user_templates = await self.get_user_templates(user_id)
        query_lower = query.lower()

        return [
            template
            for template in user_templates
            if (
                query_lower in template.name.lower()
                or (template.description and query_lower in template.description.lower())
                or query_lower in template.pattern.lower()
            )
        ]

    async def get_template_usage_stats(self, user_id: str) -> dict[str, Any]:
        """Get usage statistics for a user's templates.

        Args:
            user_id: The user identifier

        Returns:
            Dictionary with usage statistics
        """
        user_templates = await self.get_user_templates(user_id)

        if not user_templates:
            return {
                "total_templates": 0,
                "total_usage": 0,
                "most_used": None,
                "average_usage": 0.0,
            }

        total_usage = sum(t.usage_count for t in user_templates)
        most_used = max(user_templates, key=lambda t: t.usage_count)

        return {
            "total_templates": len(user_templates),
            "total_usage": total_usage,
            "most_used": {
                "id": most_used.id,
                "name": most_used.name,
                "usage_count": most_used.usage_count,
            },
            "average_usage": total_usage / len(user_templates),
            "templates": [
                {
                    "id": t.id,
                    "name": t.name,
                    "usage_count": t.usage_count,
                    "last_used": None,  # TODO: Track last_used timestamp
                }
                for t in user_templates
            ],
        }

    async def clone_template(self, template_id: str, new_user_id: str, new_name: str | None = None) -> str:
        """Clone an existing template for a new user.

        Args:
            template_id: The ID of the template to clone
            new_user_id: The user ID for the cloned template
            new_name: Optional new name (defaults to "Copy of {original_name}")

        Returns:
            The ID of the cloned template

        Raises:
            ValueError: If the template doesn't exist
        """
        original = await self.get_template(template_id)
        if not original:
            raise ValueError(f"Template {template_id} not found")

        cloned_name = new_name or f"Copy of {original.name}"

        cloned_template = NamingTemplate(
            id=str(uuid.uuid4()),
            name=cloned_name,
            pattern=original.pattern,
            user_id=new_user_id,
            created_at=datetime.now(UTC),
            usage_count=0,
            description=original.description,
            is_active=True,
        )

        return await self.save_template(cloned_template)


# Global instance for the service
template_manager = TemplateManager()
