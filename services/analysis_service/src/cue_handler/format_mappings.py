"""Format mappings and conversion rules for different CUE formats."""

from typing import Dict, List, Any, Tuple, cast
from .generator import CueFormat


# Format capability matrix
FORMAT_CAPABILITIES = {
    CueFormat.STANDARD: {
        "max_tracks": 99,
        "multi_file": True,
        "rem_fields": "all",
        "pregap_postgap": True,
        "flags": "all",
        "char_limit": 80,
        "isrc_support": True,
        "bpm_storage": "REM",
        "color_coding": False,
        "loop_points": False,
        "beat_grid": False,
    },
    CueFormat.CDJ: {
        "max_tracks": 999,
        "multi_file": False,
        "rem_fields": "limited",
        "pregap_postgap": False,
        "flags": "limited",
        "char_limit": 80,
        "isrc_support": True,
        "bpm_storage": "REM",
        "color_coding": False,
        "loop_points": False,
        "beat_grid": False,
    },
    CueFormat.TRAKTOR: {
        "max_tracks": None,  # Unlimited
        "multi_file": True,
        "rem_fields": "extended",
        "pregap_postgap": False,
        "flags": None,
        "char_limit": 255,
        "isrc_support": False,
        "bpm_storage": "native",
        "color_coding": True,
        "loop_points": True,
        "beat_grid": True,
    },
    CueFormat.SERATO: {
        "max_tracks": None,  # Unlimited
        "multi_file": False,
        "rem_fields": "custom",
        "pregap_postgap": False,
        "flags": None,
        "char_limit": 255,
        "isrc_support": False,
        "bpm_storage": "native",
        "color_coding": True,
        "loop_points": True,
        "beat_grid": True,
    },
    CueFormat.REKORDBOX: {
        "max_tracks": None,  # Unlimited
        "multi_file": False,
        "rem_fields": "custom",
        "pregap_postgap": False,
        "flags": None,
        "char_limit": 255,
        "isrc_support": True,
        "bpm_storage": "native",
        "color_coding": True,
        "loop_points": True,
        "beat_grid": True,
    },
    CueFormat.KODI: {
        "max_tracks": 99,
        "multi_file": True,
        "rem_fields": "extended",
        "pregap_postgap": True,
        "flags": "limited",
        "char_limit": 255,
        "isrc_support": True,
        "bpm_storage": "REM",
        "color_coding": False,
        "loop_points": False,
        "beat_grid": False,
    },
}


# Conversion rules between formats
CONVERSION_RULES = {
    # Standard to CDJ
    ("standard", "cdj"): {
        "remove_commands": ["PREGAP", "POSTGAP", "SONGWRITER"],
        "limit_tracks": 999,
        "force_single_file": True,
        "simplify_rem": True,
        "preserve": ["TITLE", "PERFORMER", "INDEX", "ISRC"],
    },
    # Standard to Traktor
    ("standard", "traktor"): {
        "add_rem_fields": ["BPM", "KEY", "ENERGY"],
        "convert_indices_to_cues": True,
        "add_grid_markers": True,
        "encoding": "UTF-8",
        "max_title_length": 255,
        "remove_commands": ["FLAGS", "ISRC"],
    },
    # Standard to Serato
    ("standard", "serato"): {
        "convert_to_serato_format": True,
        "add_color_tags": True,
        "generate_analysis_data": True,
        "convert_cues_to_memory": True,
        "add_flip_data": False,
        "force_single_file": True,
        "encoding": "UTF-8",
        "remove_commands": ["FLAGS", "PREGAP", "POSTGAP"],
    },
    # Standard to Rekordbox
    ("standard", "rekordbox"): {
        "generate_xml": False,  # Optional XML export
        "add_memory_cues": True,
        "add_hot_cues": True,
        "convert_beat_grid": True,
        "add_phrase_analysis": False,
        "preserve_isrc": True,
        "force_single_file": True,
        "encoding": "UTF-8",
        "remove_commands": ["FLAGS", "PREGAP", "POSTGAP"],
    },
    # Standard to Kodi
    ("standard", "kodi"): {
        "add_discnumber": True,
        "add_replay_gain": True,
        "generate_nfo": True,
        "encoding": "UTF-8",
        "preserve_all_rem": True,
        "add_rem_fields": ["DISCNUMBER", "REPLAYGAIN_TRACK_GAIN", "REPLAYGAIN_ALBUM_GAIN"],
    },
    # CDJ to Standard
    ("cdj", "standard"): {
        "restore_multi_file": False,  # Cannot restore lost multi-file info
        "add_standard_flags": True,
    },
    # Traktor to Standard
    ("traktor", "standard"): {
        "convert_cues_to_indices": True,
        "store_grid_as_rem": True,
        "store_loops_as_rem": True,
        "remove_traktor_specific": True,
    },
    # Serato to Standard
    ("serato", "standard"): {
        "convert_memory_to_indices": True,
        "store_color_as_rem": True,
        "store_loops_as_rem": True,
        "remove_serato_specific": True,
    },
    # Rekordbox to Standard
    ("rekordbox", "standard"): {
        "convert_hot_cues_to_indices": True,
        "store_beat_grid_as_rem": True,
        "store_phrase_as_rem": True,
        "preserve_isrc": True,
    },
    # Kodi to Standard
    ("kodi", "standard"): {
        "preserve_all_commands": True,
        "simplify_rem": False,
    },
    # Cross-DJ format conversions
    ("traktor", "serato"): {
        "convert_grid_format": True,
        "map_cue_colors": True,
        "convert_loop_format": True,
        "force_single_file": True,
    },
    ("traktor", "rekordbox"): {
        "convert_grid_format": True,
        "map_cue_types": True,
        "add_phrase_markers": False,
        "preserve_key_data": True,
    },
    ("serato", "rekordbox"): {
        "convert_flip_to_memory": True,
        "map_color_scheme": True,
        "convert_analysis_data": True,
        "add_isrc_if_available": True,
    },
    ("rekordbox", "traktor"): {
        "convert_memory_to_cues": True,
        "convert_phrase_to_grid": True,
        "remove_isrc": True,
        "map_energy_levels": True,
    },
    ("rekordbox", "serato"): {
        "convert_hot_cues": True,
        "map_color_scheme": True,
        "convert_beat_grid": True,
        "remove_phrase_data": True,
    },
}


# Lossy conversion warnings
LOSSY_CONVERSIONS: Dict[Tuple[str, str], List[str]] = {
    ("standard", "cdj"): [
        "PREGAP/POSTGAP commands will be removed",
        "Multi-file references will be consolidated",
        "Extended REM fields may be lost",
        "Complex FLAGS will be simplified",
    ],
    ("cdj", "standard"): [
        "CDJ-specific optimizations will be lost",
        "Cannot restore multi-file structure if consolidated",
    ],
    ("standard", "traktor"): [
        "ISRC codes not supported in Traktor",
        "FLAGS will be ignored",
        "PREGAP/POSTGAP timing will be lost",
    ],
    ("traktor", "standard"): [
        "Beat grid information will be stored as REM",
        "Color coding will be lost",
        "Loop points will be lost",
        "Energy levels stored as REM only",
    ],
    ("standard", "serato"): [
        "ISRC codes not supported in Serato",
        "FLAGS will be ignored",
        "Multi-file structure will be lost",
    ],
    ("serato", "standard"): [
        "Flip data will be lost",
        "Color coding stored as REM only",
        "Loop points stored as REM only",
        "Analysis data will be lost",
    ],
    ("standard", "rekordbox"): [
        "FLAGS will be ignored",
        "PREGAP/POSTGAP will be removed",
        "Multi-file structure will be lost",
    ],
    ("rekordbox", "standard"): [
        "Phrase analysis will be stored as REM",
        "Memory cues may lose precision",
        "Beat grid stored as REM only",
        "Hot cue colors will be lost",
    ],
    ("standard", "kodi"): [
        # Kodi preserves most standard features
        "Some advanced FLAGS may not be supported",
    ],
    ("kodi", "standard"): [
        "NFO companion files will not be converted",
        "Album art references may be lost",
    ],
    ("traktor", "serato"): [
        "Grid marker precision may change",
        "Cue point types may be simplified",
        "Energy data will be lost",
    ],
    ("serato", "traktor"): [
        "Flip data cannot be converted",
        "Serato-specific effects will be lost",
    ],
    ("traktor", "rekordbox"): [
        "Energy levels may not map perfectly",
        "Grid format differences may affect timing",
    ],
    ("rekordbox", "traktor"): [
        "Phrase analysis will be lost",
        "ISRC codes will be removed",
        "Memory cue hierarchy will be flattened",
    ],
    ("serato", "rekordbox"): [
        "Flip features have no equivalent",
        "Color mapping may not be exact",
    ],
    ("rekordbox", "serato"): [
        "Phrase analysis will be lost",
        "ISRC codes will be removed",
        "Hot cue hierarchy differences",
    ],
}


# Command mapping between formats
COMMAND_MAPPINGS = {
    "standard_to_cdj": {
        "TITLE": "TITLE",
        "PERFORMER": "PERFORMER",
        "FILE": "FILE",
        "TRACK": "TRACK",
        "INDEX": "INDEX",
        "REM": "REM",  # Limited
        "ISRC": "ISRC",
        "FLAGS": "FLAGS",  # Limited
        # Removed: PREGAP, POSTGAP, SONGWRITER
    },
    "standard_to_traktor": {
        "TITLE": "TITLE",
        "PERFORMER": "PERFORMER",
        "FILE": "FILE",
        "TRACK": "TRACK",
        "INDEX": "CUE",  # Converted
        "REM": "REM",  # Extended with BPM, KEY, etc.
        # Removed: FLAGS, ISRC, PREGAP, POSTGAP
    },
    "standard_to_serato": {
        "TITLE": "TITLE",
        "PERFORMER": "PERFORMER",
        "FILE": "FILE",
        "TRACK": "TRACK",
        "INDEX": "MEMORY_CUE",  # Converted
        "REM": "REM",  # Custom format
        # Removed: FLAGS, ISRC, PREGAP, POSTGAP
    },
    "standard_to_rekordbox": {
        "TITLE": "TITLE",
        "PERFORMER": "PERFORMER",
        "FILE": "FILE",
        "TRACK": "TRACK",
        "INDEX": "HOT_CUE",  # Converted
        "REM": "REM",  # Custom format
        "ISRC": "ISRC",
        # Removed: FLAGS, PREGAP, POSTGAP
    },
    "standard_to_kodi": {
        # Kodi preserves all standard commands
        "TITLE": "TITLE",
        "PERFORMER": "PERFORMER",
        "FILE": "FILE",
        "TRACK": "TRACK",
        "INDEX": "INDEX",
        "REM": "REM",  # Extended
        "ISRC": "ISRC",
        "FLAGS": "FLAGS",
        "PREGAP": "PREGAP",
        "POSTGAP": "POSTGAP",
    },
}


# REM field mappings
REM_FIELD_MAPPINGS = {
    "traktor": {
        "BPM": "BPM",
        "KEY": "KEY",
        "ENERGY": "ENERGY",
        "GRID": "GRID_MARKERS",
        "CUE_COLOR": "COLOR",
    },
    "serato": {
        "BPM": "BPM",
        "KEY": "KEY",
        "COLOR": "COLOR_TAG",
        "FLIP": "FLIP_DATA",
        "ANALYSIS": "ANALYSIS_VERSION",
    },
    "rekordbox": {
        "BPM": "BPM",
        "KEY": "KEY",
        "MEMORY_CUE": "MEMORY_CUE_POINT",
        "HOT_CUE": "HOT_CUE_POINT",
        "BEAT_GRID": "BEAT_GRID_DATA",
        "PHRASE": "PHRASE_ANALYSIS",
    },
    "kodi": {
        "DISCNUMBER": "DISCNUMBER",
        "REPLAYGAIN_TRACK_GAIN": "REPLAYGAIN_TRACK_GAIN",
        "REPLAYGAIN_ALBUM_GAIN": "REPLAYGAIN_ALBUM_GAIN",
        "ALBUM_ARTIST": "ALBUMARTIST",
        "COMPILATION": "COMPILATION",
    },
}


def get_format_from_string(format_str: str) -> CueFormat:
    """Convert a string to CueFormat enum.

    Args:
        format_str: Format name as string

    Returns:
        CueFormat enum value

    Raises:
        ValueError: If format is not recognized
    """
    format_str = format_str.lower().strip()
    format_map = {
        "standard": CueFormat.STANDARD,
        "cdj": CueFormat.CDJ,
        "traktor": CueFormat.TRAKTOR,
        "serato": CueFormat.SERATO,
        "rekordbox": CueFormat.REKORDBOX,
        "kodi": CueFormat.KODI,
    }

    if format_str not in format_map:
        raise ValueError(f"Unknown format: {format_str}. Supported formats: {', '.join(format_map.keys())}")

    return format_map[format_str]


def get_format_capabilities(format_type: CueFormat) -> Dict[str, Any]:
    """Get capabilities for a specific format.

    Args:
        format_type: CueFormat to get capabilities for

    Returns:
        Dictionary of format capabilities
    """
    return cast(Dict[str, Any], FORMAT_CAPABILITIES.get(format_type, {}))


def get_conversion_rules(source: CueFormat, target: CueFormat) -> Dict[str, Any]:
    """Get conversion rules between two formats.

    Args:
        source: Source format
        target: Target format

    Returns:
        Dictionary of conversion rules
    """
    key = (source.value, target.value)
    return cast(Dict[str, Any], CONVERSION_RULES.get(key, {}))


def get_lossy_warnings(source: CueFormat, target: CueFormat) -> List[str]:
    """Get warnings about lossy conversions.

    Args:
        source: Source format
        target: Target format

    Returns:
        List of warning messages
    """
    key = (source.value, target.value)
    return LOSSY_CONVERSIONS.get(key, [])
