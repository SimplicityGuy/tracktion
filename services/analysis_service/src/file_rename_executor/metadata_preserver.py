"""Metadata preservation for file rename operations."""

import logging
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from mutagen.oggvorbis import OggVorbis

logger = logging.getLogger(__name__)


class MetadataPreserver:
    """Handles metadata preservation during file rename operations."""

    @staticmethod
    def snapshot_metadata(file_path: str) -> dict[str, Any] | None:
        """Capture all metadata from an audio file.

        Supports OGG Vorbis files with full tag preservation.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary containing all metadata or None if unable to read
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return None

            # Load the file with mutagen
            audio = MutagenFile(file_path)
            if not audio:
                logger.warning(f"Unable to read audio file: {file_path}")
                return None

            metadata: dict[str, Any] = {
                "format": audio.mime[0] if audio.mime else "unknown",
                "file_path": file_path,
            }

            # Check if it's an OGG Vorbis file
            if isinstance(audio, OggVorbis):
                metadata["is_ogg"] = True

                # Save all Vorbis comments
                if audio.tags:
                    tags_dict: dict[str, Any] = {}
                    for key, values in audio.tags.items():
                        # Store all values (Vorbis comments can have multiple values per key)
                        tags_dict[str(key)] = values
                    metadata["tags"] = tags_dict

                # Save technical info
                if audio.info:
                    info_dict: dict[str, Any] = {
                        "length": audio.info.length,
                        "bitrate": audio.info.bitrate,
                        "sample_rate": audio.info.sample_rate,
                        "channels": audio.info.channels,
                    }

                    # OGG-specific technical info
                    if hasattr(audio.info, "bitrate_nominal"):
                        info_dict["bitrate_nominal"] = audio.info.bitrate_nominal
                    if hasattr(audio.info, "bitrate_lower"):
                        info_dict["bitrate_lower"] = audio.info.bitrate_lower
                    if hasattr(audio.info, "bitrate_upper"):
                        info_dict["bitrate_upper"] = audio.info.bitrate_upper
                    if hasattr(audio.info, "encoder_version"):
                        info_dict["encoder_version"] = audio.info.encoder_version

                    metadata["info"] = info_dict
            else:
                # For non-OGG files, still capture basic metadata
                metadata["is_ogg"] = False

                if hasattr(audio, "tags") and audio.tags:
                    tags_dict_general: dict[str, Any] = {}
                    # Try to extract common tags
                    for key in audio.tags:
                        try:
                            value = audio.tags[key]
                            # Convert to string if possible
                            if isinstance(value, list):
                                tags_dict_general[str(key)] = [str(v) for v in value]
                            else:
                                tags_dict_general[str(key)] = str(value)
                        except Exception as e:
                            logger.debug(f"Could not extract tag {key}: {e}")
                    metadata["tags"] = tags_dict_general

                if hasattr(audio, "info") and audio.info:
                    info_dict_general: dict[str, Any] = {}
                    # Try to extract common info fields
                    for attr in ["length", "bitrate", "sample_rate", "channels"]:
                        if hasattr(audio.info, attr):
                            info_dict_general[str(attr)] = getattr(audio.info, attr)
                    metadata["info"] = info_dict_general

            logger.debug(f"Captured metadata snapshot for {file_path}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to snapshot metadata from {file_path}: {e}")
            return None

    @staticmethod
    def restore_metadata(file_path: str, metadata: dict[str, Any]) -> bool:
        """Restore metadata to an audio file after rename.

        Args:
            file_path: Path to the renamed audio file
            metadata: Metadata dictionary to restore

        Returns:
            True if successful, False otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False

            # Check if this was an OGG file
            if not metadata.get("is_ogg", False):
                logger.debug(f"Skipping metadata restoration for non-OGG file: {file_path}")
                return True  # Not an error, just not needed

            # Load the file as OGG Vorbis
            try:
                audio = OggVorbis(file_path)
            except Exception as e:
                logger.error(f"Failed to load OGG file {file_path}: {e}")
                return False

            # Restore tags
            if "tags" in metadata and metadata["tags"] and audio.tags is not None:
                # Clear existing tags
                audio.tags.clear()

                # Restore all tags
                for key, original_values in metadata["tags"].items():
                    # Ensure values is a list (Vorbis comments require lists)
                    processed_values = original_values
                    if not isinstance(processed_values, list):
                        processed_values = [processed_values]
                    audio.tags[key] = processed_values

                # Save the changes
                audio.save()
                logger.info(f"Successfully restored metadata to {file_path}")
                return True
            logger.warning(f"No tags to restore for {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore metadata to {file_path}: {e}")
            return False

    @staticmethod
    def verify_metadata(original_path: str, new_path: str, metadata_snapshot: dict[str, Any]) -> bool:
        """Verify that metadata was preserved correctly after rename.

        Args:
            original_path: Original file path (for reference)
            new_path: New file path after rename
            metadata_snapshot: Original metadata snapshot

        Returns:
            True if metadata matches, False otherwise
        """
        try:
            # Capture current metadata
            current_metadata = MetadataPreserver.snapshot_metadata(new_path)
            if not current_metadata:
                logger.error(f"Could not read metadata from renamed file: {new_path}")
                return False

            # For OGG files, verify tags match
            if metadata_snapshot.get("is_ogg", False):
                original_tags = metadata_snapshot.get("tags", {})
                current_tags = current_metadata.get("tags", {})

                # Compare tag counts
                if len(original_tags) != len(current_tags):
                    logger.warning(f"Tag count mismatch: original={len(original_tags)}, current={len(current_tags)}")
                    return False

                # Compare each tag
                for key, original_values in original_tags.items():
                    if key not in current_tags:
                        logger.warning(f"Missing tag after rename: {key}")
                        return False

                    current_values = current_tags[key]
                    if original_values != current_values:
                        logger.warning(
                            f"Tag value mismatch for {key}: original={original_values}, current={current_values}"
                        )
                        return False

                # Check technical info (should remain the same)
                original_info = metadata_snapshot.get("info", {})
                current_info = current_metadata.get("info", {})

                # Compare key technical fields
                for field in ["length", "bitrate", "sample_rate", "channels"]:
                    if field in original_info and original_info[field] != current_info.get(field):
                        logger.warning(
                            f"Technical info mismatch for {field}: "
                            f"original={original_info[field]}, "
                            f"current={current_info.get(field)}"
                        )
                        # Technical info changes are less critical

            logger.info(f"Metadata verification successful for {new_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to verify metadata: {e}")
            return False

    @staticmethod
    def format_metadata_for_display(metadata: dict[str, Any]) -> str:
        """Format metadata dictionary for human-readable display.

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted string representation
        """
        if metadata is None:
            return "No metadata"

        lines = []

        # Format header
        lines.append(f"Format: {metadata.get('format', 'unknown')}")
        if metadata.get("is_ogg"):
            lines.append("Type: OGG Vorbis")

        # Format tags
        if metadata.get("tags"):
            lines.append("\nTags:")
            for key, values in sorted(metadata["tags"].items()):
                if isinstance(values, list):
                    lines.extend(f"  {key}: {value}" for value in values)
                else:
                    lines.append(f"  {key}: {values}")

        # Format technical info
        if metadata.get("info"):
            lines.append("\nTechnical Info:")
            info = metadata["info"]
            if "length" in info:
                lines.append(f"  Duration: {info['length']:.2f} seconds")
            if "bitrate" in info:
                lines.append(f"  Bitrate: {info['bitrate']} bps")
            if "sample_rate" in info:
                lines.append(f"  Sample Rate: {info['sample_rate']} Hz")
            if "channels" in info:
                lines.append(f"  Channels: {info['channels']}")
            if "bitrate_nominal" in info:
                lines.append(f"  Nominal Bitrate: {info['bitrate_nominal']} bps")

        return "\n".join(lines)
