"""Tests for Discord message templates."""

from services.notification_service.src.core.base import AlertType
from services.notification_service.src.templates.discord_templates import (
    DiscordEmbedBuilder,
)


class TestDiscordEmbedBuilder:
    """Test DiscordEmbedBuilder functionality."""

    def test_basic_embed(self) -> None:
        """Test building a basic embed."""
        builder = DiscordEmbedBuilder()

        embed = builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="Test Title",
            description="Test description",
        )

        assert "embeds" in embed
        assert len(embed["embeds"]) == 1

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "Test Title"
        assert embed_data["description"] == "Test description"
        assert embed_data["color"] == 0x3498DB  # Blue for GENERAL
        assert "timestamp" in embed_data
        assert embed_data["footer"]["text"] == "Tracktion General Alert"

    def test_embed_with_fields(self) -> None:
        """Test building embed with fields."""
        builder = DiscordEmbedBuilder()

        fields = [
            {"name": "Field 1", "value": "Value 1", "inline": True},
            {"name": "Field 2", "value": "Value 2", "inline": False},
        ]

        embed = builder.build_embed(
            alert_type=AlertType.ERROR,
            title="Error Alert",
            description="Something went wrong",
            fields=fields,
        )

        embed_data = embed["embeds"][0]
        assert embed_data["color"] == 0xE67E22  # Orange for ERROR
        assert len(embed_data["fields"]) == 2
        assert embed_data["fields"][0]["name"] == "Field 1"
        assert embed_data["fields"][0]["inline"] is True
        assert embed_data["fields"][1]["inline"] is False

    def test_embed_with_custom_color(self) -> None:
        """Test building embed with custom color override."""
        builder = DiscordEmbedBuilder()

        embed = builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="Custom Color",
            description="Custom color test",
            color=0xFF0000,  # Red
        )

        embed_data = embed["embeds"][0]
        assert embed_data["color"] == 0xFF0000

    def test_embed_with_all_options(self) -> None:
        """Test building embed with all optional parameters."""
        builder = DiscordEmbedBuilder()

        fields = [{"name": "Test Field", "value": "Test Value"}]
        author = {"name": "Test Author", "url": "https://example.com"}
        footer = {"text": "Custom Footer", "icon_url": "https://example.com/icon.png"}

        embed = builder.build_embed(
            alert_type=AlertType.CRITICAL,
            title="Full Featured Embed",
            description="Testing all options",
            fields=fields,
            url="https://example.com",
            thumbnail_url="https://example.com/thumb.png",
            image_url="https://example.com/image.png",
            author=author,
            footer=footer,
        )

        embed_data = embed["embeds"][0]
        assert embed_data["url"] == "https://example.com"
        assert embed_data["thumbnail"]["url"] == "https://example.com/thumb.png"
        assert embed_data["image"]["url"] == "https://example.com/image.png"
        assert embed_data["author"] == author
        assert embed_data["footer"] == footer

    def test_color_mapping(self) -> None:
        """Test that alert types map to correct colors."""
        builder = DiscordEmbedBuilder()

        color_tests = [
            (AlertType.GENERAL, 0x3498DB),  # Blue
            (AlertType.ERROR, 0xE67E22),  # Orange
            (AlertType.CRITICAL, 0xE74C3C),  # Red
            (AlertType.TRACKLIST, 0x2ECC71),  # Green
            (AlertType.MONITORING, 0x9B59B6),  # Purple
            (AlertType.SECURITY, 0xF39C12),  # Yellow
        ]

        for alert_type, expected_color in color_tests:
            embed = builder.build_embed(
                alert_type=alert_type,
                title="Color Test",
                description="Testing colors",
            )
            assert embed["embeds"][0]["color"] == expected_color

    def test_title_truncation(self) -> None:
        """Test that long titles are truncated."""
        builder = DiscordEmbedBuilder()

        long_title = "A" * 300  # Longer than 256 char limit
        truncated = builder._truncate_title(long_title)

        assert len(truncated) <= 256
        assert truncated.endswith("...")

    def test_description_truncation(self) -> None:
        """Test that long descriptions are truncated."""
        builder = DiscordEmbedBuilder()

        long_description = "B" * 5000  # Longer than 4096 char limit
        truncated = builder._truncate_description(long_description)

        assert len(truncated) <= 4096
        assert truncated.endswith("...")

    def test_fields_formatting(self) -> None:
        """Test that fields are properly formatted and limited."""
        builder = DiscordEmbedBuilder()

        # Create more than 25 fields (Discord limit)
        many_fields = [{"name": f"Field {i}", "value": f"Value {i}"} for i in range(30)]

        formatted = builder._format_fields(many_fields)

        # Should be limited to 25
        assert len(formatted) == 25

        # Check formatting
        for field in formatted:
            assert "name" in field
            assert "value" in field
            assert "inline" in field

    def test_error_embed(self) -> None:
        """Test building error notification embed."""
        builder = DiscordEmbedBuilder()

        error_details = {
            "component": "parser",
            "error_code": "PARSE_001",
            "file_path": "/path/to/file.txt",
        }

        embed = builder.build_error_embed(
            error_message="Failed to parse file",
            error_details=error_details,
            traceback="Traceback (most recent call last):\n  File test.py, line 1",
        )

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "❌ Error Occurred"
        assert embed_data["description"] == "Failed to parse file"
        assert embed_data["color"] == 0xE67E22  # Error color

        # Check fields
        field_names = [field["name"] for field in embed_data["fields"]]
        assert "Component" in field_names
        assert "Error Code" in field_names
        assert "File Path" in field_names
        assert "Traceback" in field_names

    def test_critical_embed(self) -> None:
        """Test building critical alert embed."""
        builder = DiscordEmbedBuilder()

        embed = builder.build_critical_embed(
            title="Database Failure",
            message="Primary database is unreachable",
            impact="All services affected",
            action_required="Restart database server",
        )

        assert "content" in embed  # Should include @here mention
        assert embed["content"] == "@here Critical Alert!"

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "🚨 Database Failure"
        assert embed_data["color"] == 0xE74C3C  # Critical color

        # Check fields
        field_names = [field["name"] for field in embed_data["fields"]]
        assert "🎯 Impact" in field_names
        assert "⚡ Action Required" in field_names

    def test_tracklist_embed(self) -> None:
        """Test building tracklist notification embed."""
        builder = DiscordEmbedBuilder()

        tracklist_info = {
            "name": "Electronic Mix #1",
            "id": "TL-001",
            "source": "radio_show",
            "stats": {"tracks": 15, "duration": "62.5 minutes"},
        }

        changes = ["Added 5 new tracks", "Updated metadata", "Fixed timing issues"]

        embed = builder.build_tracklist_embed(
            action="updated",
            tracklist_info=tracklist_info,
            changes=changes,
        )

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "📋 Tracklist Updated"
        assert "Electronic Mix #1" in embed_data["description"]
        assert embed_data["color"] == 0x2ECC71  # Tracklist color

        # Check fields
        field_names = [field["name"] for field in embed_data["fields"]]
        assert "📝 Changes" in field_names
        assert "📊 Statistics" in field_names

    def test_monitoring_embed(self) -> None:
        """Test building monitoring alert embed."""
        builder = DiscordEmbedBuilder()

        embed = builder.build_monitoring_embed(
            metric_name="CPU Usage",
            current_value="95%",
            threshold="80%",
            status="error",
        )

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "🔴 Monitoring Alert: CPU Usage"
        assert "**Current Value:** 95%" in embed_data["description"]
        assert "**Threshold:** 80%" in embed_data["description"]
        assert embed_data["color"] == 0x9B59B6  # Monitoring color

    def test_security_embed(self) -> None:
        """Test building security alert embed."""
        builder = DiscordEmbedBuilder()

        details = {
            "source_ip": "192.168.1.100",
            "user_agent": "suspicious-bot",
            "attempts": 5,
        }

        embed = builder.build_security_embed(
            security_event="Failed Login Attempts",
            details=details,
            severity="high",
        )

        embed_data = embed["embeds"][0]
        assert embed_data["title"] == "🟠 Security Alert: Failed Login Attempts"
        assert embed_data["color"] == 0xF39C12  # Security color

        # Check fields
        field_names = [field["name"] for field in embed_data["fields"]]
        assert "Source Ip" in field_names
        assert "Severity" in field_names

    def test_stats_formatting(self) -> None:
        """Test statistics formatting."""
        builder = DiscordEmbedBuilder()

        stats = {
            "track_count": 15,
            "total_duration": 3750.5,  # seconds
            "average_bpm": 128.7,
            "file_size": 1024000,  # bytes
        }

        formatted = builder._format_stats(stats)

        assert "**Track Count:** 15" in formatted
        assert "**Total Duration:** 3750.50" in formatted
        assert "**Average Bpm:** 128.70" in formatted

    def test_field_value_truncation(self) -> None:
        """Test that field values are truncated to Discord limits."""
        builder = DiscordEmbedBuilder()

        long_value = "X" * 2000  # Longer than 1024 limit
        fields = [{"name": "Test Field", "value": long_value}]

        formatted = builder._format_fields(fields)

        assert len(formatted[0]["value"]) <= 1024

    def test_footer_icon_url(self) -> None:
        """Test footer icon URL generation."""
        builder = DiscordEmbedBuilder()

        # All alert types should return a valid URL
        for alert_type in AlertType:
            icon_url = builder._get_footer_icon(alert_type)
            assert icon_url.startswith("https://")

    def test_status_emoji_mapping(self) -> None:
        """Test status emoji mapping in monitoring alerts."""
        builder = DiscordEmbedBuilder()

        status_tests = [
            ("info", "🔵"),  # Updated to match the new emoji
            ("warning", "⚠️"),
            ("error", "🔴"),
            ("success", "✅"),
            ("unknown", "📊"),  # Default
        ]

        for status, expected_emoji in status_tests:
            embed = builder.build_monitoring_embed(
                metric_name="Test Metric",
                current_value="100",
                status=status,
            )

            title = embed["embeds"][0]["title"]
            assert title.startswith(expected_emoji)

    def test_severity_emoji_mapping(self) -> None:
        """Test severity emoji mapping in security alerts."""
        builder = DiscordEmbedBuilder()

        severity_tests = [
            ("low", "🟢"),
            ("medium", "🟡"),
            ("high", "🟠"),
            ("critical", "🔴"),
            ("unknown", "⚠️"),  # Default
        ]

        for severity, expected_emoji in severity_tests:
            embed = builder.build_security_embed(
                security_event="Test Event",
                details={"test": "value"},
                severity=severity,
            )

            title = embed["embeds"][0]["title"]
            assert title.startswith(expected_emoji)

    def test_changes_truncation(self) -> None:
        """Test that changes list is truncated in tracklist embed."""
        builder = DiscordEmbedBuilder()

        # Create more than 10 changes
        many_changes = [f"Change {i}" for i in range(15)]

        embed = builder.build_tracklist_embed(
            action="updated",
            tracklist_info={"name": "Test Tracklist"},
            changes=many_changes,
        )

        # Find changes field
        changes_field = None
        for field in embed["embeds"][0]["fields"]:
            if field["name"] == "📝 Changes":
                changes_field = field
                break

        assert changes_field is not None
        # Should show "... and 5 more"
        assert "... and 5 more" in changes_field["value"]
