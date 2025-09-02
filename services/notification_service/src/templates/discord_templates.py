"""Discord message templates and builders."""

from datetime import UTC, datetime
from typing import Any, ClassVar

from services.notification_service.src.core.base import AlertType


class DiscordEmbedBuilder:
    """Builder for Discord embed messages."""

    # Discord color codes for different alert types
    COLOR_MAP: ClassVar[dict[AlertType, int]] = {
        AlertType.GENERAL: 0x3498DB,  # Blue
        AlertType.ERROR: 0xE67E22,  # Orange
        AlertType.CRITICAL: 0xE74C3C,  # Red
        AlertType.TRACKLIST: 0x2ECC71,  # Green
        AlertType.MONITORING: 0x9B59B6,  # Purple
        AlertType.SECURITY: 0xF39C12,  # Yellow
    }

    # Default color for unknown alert types
    DEFAULT_COLOR: ClassVar[int] = 0x95A5A6  # Gray

    def build_embed(
        self,
        alert_type: AlertType,
        title: str,
        description: str,
        color: int | None = None,
        fields: list[dict[str, Any]] | None = None,
        url: str | None = None,
        thumbnail_url: str | None = None,
        image_url: str | None = None,
        author: dict[str, str] | None = None,
        footer: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build a Discord embed message.

        Args:
            alert_type: Type of alert
            title: Embed title
            description: Embed description
            color: Optional color override
            fields: Optional list of fields
            url: Optional URL for the title
            thumbnail_url: Optional thumbnail image URL
            image_url: Optional main image URL
            author: Optional author information
            footer: Optional footer information

        Returns:
            Discord webhook payload with embed
        """
        # Build embed object
        embed: dict[str, Any] = {
            "title": self._truncate_title(title),
            "description": self._truncate_description(description),
            "color": color or self.COLOR_MAP.get(alert_type, self.DEFAULT_COLOR),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Add optional URL
        if url:
            embed["url"] = url

        # Add fields if provided
        if fields:
            embed["fields"] = self._format_fields(fields)

        # Add thumbnail if provided
        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}

        # Add image if provided
        if image_url:
            embed["image"] = {"url": image_url}

        # Add author if provided
        if author:
            embed["author"] = author

        # Add footer (default or custom)
        if footer:
            embed["footer"] = footer
        else:
            embed["footer"] = {
                "text": f"Tracktion {alert_type.value.title()} Alert",
                "icon_url": self._get_footer_icon(alert_type),
            }

        return {"embeds": [embed]}

    def build_error_embed(
        self,
        error_message: str,
        error_details: dict[str, Any] | None = None,
        traceback: str | None = None,
    ) -> dict[str, Any]:
        """Build an error notification embed.

        Args:
            error_message: Main error message
            error_details: Optional error details
            traceback: Optional traceback string

        Returns:
            Discord webhook payload with error embed
        """
        fields = []

        if error_details:
            for key, value in error_details.items():
                fields.append(
                    {
                        "name": key.replace("_", " ").title(),
                        "value": str(value)[:1024],  # Discord field limit
                        "inline": True,
                    }
                )

        if traceback:
            # Truncate traceback to fit Discord limits
            truncated_tb = traceback[-1000:] if len(traceback) > 1000 else traceback
            fields.append(
                {
                    "name": "Traceback",
                    "value": f"```python\n{truncated_tb}\n```",
                    "inline": False,
                }
            )

        return self.build_embed(
            alert_type=AlertType.ERROR,
            title="❌ Error Occurred",
            description=error_message,
            fields=fields,
        )

    def build_critical_embed(
        self,
        title: str,
        message: str,
        impact: str | None = None,
        action_required: str | None = None,
    ) -> dict[str, Any]:
        """Build a critical alert embed.

        Args:
            title: Alert title
            message: Alert message
            impact: Optional impact description
            action_required: Optional action required

        Returns:
            Discord webhook payload with critical embed
        """
        fields = []

        if impact:
            fields.append(
                {
                    "name": "🎯 Impact",
                    "value": impact,
                    "inline": False,
                }
            )

        if action_required:
            fields.append(
                {
                    "name": "⚡ Action Required",
                    "value": action_required,
                    "inline": False,
                }
            )

        # Add mention for critical alerts
        content = "@here Critical Alert!"

        embed = self.build_embed(
            alert_type=AlertType.CRITICAL,
            title=f"🚨 {title}",
            description=message,
            fields=fields,
        )

        # Add content for mentions
        embed["content"] = content

        return embed

    def build_tracklist_embed(
        self,
        action: str,
        tracklist_info: dict[str, Any],
        changes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a tracklist notification embed.

        Args:
            action: Action performed (e.g., "created", "updated", "deleted")
            tracklist_info: Tracklist information
            changes: Optional list of changes made

        Returns:
            Discord webhook payload with tracklist embed
        """
        title = f"📋 Tracklist {action.title()}"

        # Build description
        description_parts = []
        if "name" in tracklist_info:
            description_parts.append(f"**Name:** {tracklist_info['name']}")
        if "id" in tracklist_info:
            description_parts.append(f"**ID:** {tracklist_info['id']}")
        if "source" in tracklist_info:
            description_parts.append(f"**Source:** {tracklist_info['source']}")

        description = "\n".join(description_parts) if description_parts else f"Tracklist {action}"

        # Build fields
        fields = []

        if changes:
            changes_text = "\n".join(f"• {change}" for change in changes[:10])
            if len(changes) > 10:
                changes_text += f"\n... and {len(changes) - 10} more"

            fields.append(
                {
                    "name": "📝 Changes",
                    "value": changes_text,
                    "inline": False,
                }
            )

        # Add statistics if available
        if "stats" in tracklist_info:
            stats = tracklist_info["stats"]
            fields.append(
                {
                    "name": "📊 Statistics",
                    "value": self._format_stats(stats),
                    "inline": True,
                }
            )

        return self.build_embed(
            alert_type=AlertType.TRACKLIST,
            title=title,
            description=description,
            fields=fields,
        )

    def build_monitoring_embed(
        self,
        metric_name: str,
        current_value: Any,
        threshold: Any | None = None,
        status: str = "warning",
    ) -> dict[str, Any]:
        """Build a monitoring alert embed.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            threshold: Optional threshold value
            status: Status level (info, warning, error)

        Returns:
            Discord webhook payload with monitoring embed
        """
        # Choose emoji based on status
        status_emoji = {
            "info": "🔵",  # Blue circle instead of info emoji
            "warning": "⚠️",
            "error": "🔴",
            "success": "✅",
        }.get(status, "📊")

        title = f"{status_emoji} Monitoring Alert: {metric_name}"

        description_parts = [f"**Current Value:** {current_value}"]
        if threshold:
            description_parts.append(f"**Threshold:** {threshold}")

        return self.build_embed(
            alert_type=AlertType.MONITORING,
            title=title,
            description="\n".join(description_parts),
        )

    def build_security_embed(
        self,
        security_event: str,
        details: dict[str, Any],
        severity: str = "medium",
    ) -> dict[str, Any]:
        """Build a security alert embed.

        Args:
            security_event: Type of security event
            details: Event details
            severity: Severity level (low, medium, high, critical)

        Returns:
            Discord webhook payload with security embed
        """
        severity_emoji = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }.get(severity, "⚠️")

        title = f"{severity_emoji} Security Alert: {security_event}"

        fields = []
        for key, value in details.items():
            fields.append(
                {
                    "name": key.replace("_", " ").title(),
                    "value": str(value)[:1024],
                    "inline": True,
                }
            )

        # Add severity field
        fields.append(
            {
                "name": "Severity",
                "value": severity.upper(),
                "inline": True,
            }
        )

        return self.build_embed(
            alert_type=AlertType.SECURITY,
            title=title,
            description="A security event has been detected",
            fields=fields,
        )

    def _truncate_title(self, title: str, max_length: int = 256) -> str:
        """Truncate title to Discord's limit.

        Args:
            title: Title to truncate
            max_length: Maximum length (Discord limit is 256)

        Returns:
            Truncated title
        """
        if len(title) <= max_length:
            return title
        return title[: max_length - 3] + "..."

    def _truncate_description(self, description: str, max_length: int = 4096) -> str:
        """Truncate description to Discord's limit.

        Args:
            description: Description to truncate
            max_length: Maximum length (Discord limit is 4096)

        Returns:
            Truncated description
        """
        if len(description) <= max_length:
            return description
        return description[: max_length - 3] + "..."

    def _format_fields(self, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format and validate fields for Discord.

        Args:
            fields: List of field dictionaries

        Returns:
            Formatted fields (max 25 fields per embed)
        """
        formatted = []
        for field in fields[:25]:  # Discord limit is 25 fields
            formatted_field = {
                "name": self._truncate_title(str(field.get("name", "Field")), 256),
                "value": self._truncate_description(str(field.get("value", "")), 1024),
                "inline": field.get("inline", True),
            }
            formatted.append(formatted_field)
        return formatted

    def _format_stats(self, stats: dict[str, Any]) -> str:
        """Format statistics for display.

        Args:
            stats: Statistics dictionary

        Returns:
            Formatted statistics string
        """
        lines = []
        for key, value in stats.items():
            # Format key
            formatted_key = key.replace("_", " ").title()
            # Format value
            formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
            lines.append(f"**{formatted_key}:** {formatted_value}")

        return "\n".join(lines[:10])  # Limit to 10 stats

    def _get_footer_icon(self, alert_type: AlertType) -> str:
        """Get footer icon URL for alert type.

        Args:
            alert_type: Type of alert

        Returns:
            Icon URL (using Discord's default icons)
        """
        # You can customize these with your own icon URLs
        return "https://cdn.discordapp.com/embed/avatars/0.png"
