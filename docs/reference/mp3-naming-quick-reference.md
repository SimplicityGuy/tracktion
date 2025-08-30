# MP3 Naming Convention Quick Reference

## 🎯 Essential Rules - Quick Lookup

### Directory Name Format
```
Artist-Title-[TAGS]-SOURCE-YEAR-GROUP
```

### Filename Format
```
TrackNumber-Artist-Song_Title.mp3
```

### Valid Characters Only
```
a-z A-Z 0-9 _ . - ( )
```

### Maximum Length
```
dirname + filename ≤ 255 characters
```

---

## 📁 Directory Examples

### Standard Formats
```bash
# Regular Album
Pink_Floyd-Dark_Side_of_the_Moon-CD-1973-GROUP

# Single
Daft_Punk-One_More_Time-CDS-2000-GROUP

# Various Artists
VA-Now_Thats_What_I_Call_Music_100-2CD-2018-GROUP

# Soundtrack
VA-Pulp_Fiction-OST-CD-1994-GROUP

# Web Single
The_Weeknd-Blinding_Lights-SINGLE-WEB-2019-GROUP

# Live Recording
Radiohead-Live_at_Glastonbury-FM-2017-GROUP

# Bootleg
The_Beatles-Studio_Outtakes-BOOTLEG-VINYL-1969-GROUP

# Multi-Disc
Queen-Greatest_Hits-3CD-2011-GROUP
```

---

## 📄 Filename Examples

### Standard Tracks
```bash
01-artist_name-song_title.mp3
02-artist_name-another_song_(remix).mp3
03-artist_name-featuring_track_ft_guest.mp3
```

### Various Artists Tracks
```bash
01-first_artist-their_song.mp3
02-second_artist-different_track.mp3
03-third_artist_ft_guest-collaboration.mp3
```

---

## 🏷️ Source Tags

### Physical Media
| Media | Required Tag | Multi-Disc |
|-------|-------------|------------|
| CD Album | none or `-CD-` | `-2CD-`, `-3CD-` |
| CD Single | `-CDS-` or `-CDM-` | `-2CDS-` |
| Vinyl Album | `-LP-` or `-VINYL-` | `-2LP-` |
| Vinyl Single | `-VLS-` or `-VINYL-` | `-2VLS-` |
| DVD | `-DVD-` | `-2DVD-` |
| Tape | `-TAPE-` | `-2TAPE-` |

### Digital Sources
| Source | Tag | Notes |
|--------|-----|-------|
| Web Purchase | `-WEB-` | Add `-SINGLE-` if one track |
| MP3 CD | `-MP3CD-` | MP3 files on data CD |
| Flash Media | `-FLASH-` | SlotMusic, etc. |

### Live Sources
| Source | Tag | Description |
|--------|-----|-------------|
| FM Radio | `-FM-` | Terrestrial radio |
| Satellite | `-SAT-` | Satellite radio |
| Cable | `-CABLE-` | Cable broadcast |
| Soundboard | `-SBD-` | Direct from mixer |
| Audience | `-AUD-` | Audience recording |
| Line-In | `-LINE-` | Direct line recording |

---

## 🎭 Special Tags

### Required Special Tags
| Scenario | Tag | Position | Example |
|----------|-----|----------|---------|
| Various Artists | `VA` | Artist position | `VA-Title-...` |
| Bootleg | `BOOTLEG` | After title | `Artist-Title-BOOTLEG-...` |
| Soundtrack | `-OST-` | After title | `Artist-Title-OST-...` |
| Web Single | `-SINGLE-` | After title | `Artist-Title-SINGLE-WEB-...` |
| Magazine CD | `-MAG-` | After source | `Artist-Title-MAG-...` |
| Internal | `_INT` | End of dirname | `Artist-Title-2021-GROUP_INT` |

---

## 🔤 Character Replacement

### Common Replacements
```python
' ' → '_'     # Space to underscore
'&' → '_and_' # Ampersand
'@' → '_at_'  # At symbol
'/' → '_'     # Slash
':' → '_'     # Colon
'?' → '_'     # Question mark
'!' → '_'     # Exclamation
',' → '_'     # Comma
'"' → '_'     # Quotes
"'" → '_'     # Apostrophe
```

### Examples
```bash
# Original: "Rock & Roll"
# Becomes:  "Rock_and_Roll"

# Original: "What's Up?"
# Becomes:  "Whats_Up"

# Original: "Live @ Madison Square"
# Becomes:  "Live_at_Madison_Square"
```

---

## 📏 Length Management

### Priority When Truncating
1. Keep track numbers intact
2. Keep source tag intact
3. Keep year intact
4. Truncate title before artist
5. Remove featuring info if needed
6. Use abbreviations as last resort

### Smart Truncation Examples
```bash
# Too Long:
Very_Long_Artist_Name_Featuring_Many_People-Extended_Album_Title_Deluxe_Edition-CD-2021-GROUP

# Truncated:
Very_Long_Artist_Name_Feat-Extended_Album_Title_Deluxe-CD-2021-GROUP

# Still Too Long:
Very_Long_Artist-Extended_Album-CD-2021-GROUP
```

---

## ✅ Validation Checklist

### Quick Validation Steps
- [ ] **Characters**: Only `a-z A-Z 0-9 _ . - ( )`
- [ ] **Length**: Total path ≤ 255 chars
- [ ] **Dirname**: Has artist, title, year, group
- [ ] **Filename**: Starts with track number
- [ ] **Source**: Valid source tag present
- [ ] **VA**: Uses "VA" for various artists
- [ ] **Special**: OST, SINGLE, BOOTLEG tags where needed

---

## 🚫 Common Mistakes

### ❌ WRONG → ✅ CORRECT

```bash
# Missing source tag
❌ Artist-Album-2021-GROUP
✅ Artist-Album-CD-2021-GROUP

# Wrong VA format
❌ Various_Artists-Compilation-2021-GROUP
✅ VA-Compilation-2021-GROUP

# Invalid characters
❌ Artist's "Best" Songs!-CD-2021-GROUP
✅ Artists_Best_Songs-CD-2021-GROUP

# Wrong track number format
❌ artist-01-song.mp3
✅ 01-artist-song.mp3

# Missing SINGLE tag for web single
❌ Artist-Song-WEB-2021-GROUP
✅ Artist-Song-SINGLE-WEB-2021-GROUP

# Wrong multi-disc format
❌ Artist-Album-CD1-2021-GROUP
✅ Artist-Album-2CD-2021-GROUP
```

---

## 🛠️ Quick Implementation

### Python Function Template
```python
def format_mp3_dirname(artist, title, source, year, group="GROUP", tags=None):
    """Quick MP3 dirname generator."""
    # Sanitize inputs
    artist = sanitize(artist)
    title = sanitize(title)

    # Build dirname
    parts = [artist, title]
    if tags:
        parts.extend(tags)
    parts.extend([source, str(year), group])

    dirname = "-".join(parts)

    # Check length
    if len(dirname) > 255:
        dirname = truncate_smart(dirname)

    return dirname

def sanitize(text):
    """Replace invalid characters."""
    # Replace spaces and invalid chars
    text = text.replace(" ", "_")
    text = text.replace("&", "_and_")
    # ... more replacements
    return text
```

### Regex Patterns
```python
# Valid dirname pattern
DIRNAME_PATTERN = r'^[A-Za-z0-9_\-()]+\-[A-Za-z0-9_\-()]+\-\d{4}\-[A-Za-z0-9_\-()]+$'

# Valid filename pattern
FILENAME_PATTERN = r'^\d{2,3}\-[A-Za-z0-9_\-()]+\-[A-Za-z0-9_\-()]+\.mp3$'

# Source tag pattern
SOURCE_PATTERN = r'\-(CD|CDS|CDM|VINYL|VLS|LP|WEB|FM|SAT|CABLE|DVD|TAPE|FLASH|MP3CD|DAB|SBD|AUD|LINE|STREAM)\-'
```

---

## 📊 Decision Trees

### Is it Various Artists?
```
Multiple artists on album?
├─ YES → Use "VA" as artist
│   └─ Include individual artists in filenames
└─ NO → Use album artist
```

### Does it need SINGLE tag?
```
Is source WEB?
├─ YES → Is it one track?
│   ├─ YES → Is it < 25 minutes?
│   │   ├─ YES → Add -SINGLE- tag
│   │   └─ NO → No SINGLE tag needed
│   └─ NO → No SINGLE tag needed
└─ NO → No SINGLE tag needed
```

### Multi-disc notation?
```
Multiple discs?
├─ YES → Add count before source
│   └─ Format: -2CD-, -3DVD-, etc.
└─ NO → Use single source tag
```

---

## 🔗 Quick Links

- [Full MP3 Rules Documentation](./mp3-naming-rules-v4.1.md)
- [Implementation Guide](./mp3-naming-implementation-guide.md)
- [Epic 12: MP3 Naming Convention](../prd/epic-12-mp3-naming-conventions.md)
