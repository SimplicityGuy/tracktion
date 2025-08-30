# MP3 Naming Convention Edge Cases & Special Scenarios

## Overview

This document covers edge cases, special scenarios, and complex situations that may arise when implementing MP3 Release Rules 4.1 naming conventions.

## Edge Case Categories

### 1. Character Handling Edge Cases

#### Foreign Characters & Accents
```bash
# Scandinavian Characters
Original: Björk - Jóga
Solution: Bjork-Joga
Rule: Remove diacritics, convert to ASCII

# German Umlauts
Original: Mötley Crüe - Fräulein
Solution: Motley_Crue-Fraulein
Alternative: Moetley_Cruee-Fraeulein (if preserving pronunciation)

# Cyrillic
Original: Тату - Нас не догонят
Solution: Tatu-Nas_Ne_Dogonyat
Rule: Transliterate to Latin alphabet

# Japanese
Original: 坂本龍一 - Energy Flow
Solution: Sakamoto_Ryuichi-Energy_Flow
Rule: Use romanization (Romaji/Hepburn)

# Arabic
Original: عمرو دياب - حبيبي
Solution: Amr_Diab-Habibi
Rule: Use common transliteration
```

#### Special Punctuation
```bash
# Apostrophes and Quotes
Original: Don't Stop Believin'
Solution: Dont_Stop_Believin

# Exclamation and Question Marks
Original: Help! / What's Going On?
Solution: Help / Whats_Going_On

# Colons and Semicolons
Original: The Matrix: Reloaded
Solution: The_Matrix_Reloaded

# Slashes
Original: AC/DC - Back in Black
Solution: AC_DC-Back_in_Black

# Parentheses (allowed but context matters)
Original: Song (Remix) [Extended]
Solution: Song_(Remix)_Extended
```

### 2. Length Limit Edge Cases

#### Extremely Long Names
```bash
# Case 1: Very long artist name
Original: "The Incredible String Band Featuring Multiple Guest Artists Including Famous People"
Problem: Exceeds 255 when combined with album and other tags

Solution Strategy:
1. First attempt: Use common abbreviations
   The_Incredible_String_Band_Feat_Multiple_Guests
2. If still too long: Truncate featuring info
   The_Incredible_String_Band
3. Last resort: Use initialism
   TISB-Album_Title-CD-2021-GROUP

# Case 2: Long album title with subtitle
Original: "The Complete Collection of Everything Ever Recorded Including Unreleased Material (Deluxe Edition) [Remastered]"

Solution Strategy:
1. Remove marketing terms
   The_Complete_Collection_Including_Unreleased
2. Further truncation
   The_Complete_Collection
3. Use abbreviation
   Complete_Collection
```

#### Multi-Disc Sets with Long Names
```bash
# Problem: Each disc adds to path length
Original: Artist_With_Long_Name-Extended_Album_Title_Special_Edition-10CD-2021-GROUP/CD01/01-artist_with_long_name-very_long_song_title_with_featuring.mp3

Solution:
1. Shorten at directory level
   Artist-Extended_Album-10CD-2021-GROUP/
2. Use disc folders efficiently
   CD01/ (not Disc_01/ or CD_01/)
3. Abbreviate in filenames
   01-artist-long_song_feat.mp3
```

### 3. Source Detection Edge Cases

#### Ambiguous Sources
```bash
# Digital release also available on CD
Scenario: Album available on both WEB and CD
Solution: Use actual source of YOUR copy
Example: If downloaded → -WEB-, If ripped from CD → -CD-

# Streaming rip vs official download
Scenario: Spotify rip vs iTunes purchase
Rule: Streaming rips are INTERNAL only (-STREAM-)
Official: Artist-Album-WEB-2021-GROUP
Internal: Artist-Album-STREAM-2021-GROUP_INT

# Hybrid releases
Scenario: CD+DVD combo pack
Solution: Release separately by source
Artist-Album-CD-2021-GROUP
Artist-Album-DVD-2021-GROUP
```

#### Rare Sources
```bash
# MiniDisc
Format: Artist-Album-MD-2021-GROUP

# DAT (Digital Audio Tape)
Format: Artist-Album-DAT-2021-GROUP

# SACD (Super Audio CD)
Format: Artist-Album-SACD-2021-GROUP

# HD formats
Format: Artist-Album-HDDVD-2021-GROUP
Format: Artist-Album-BD-2021-GROUP
```

### 4. Various Artists Edge Cases

#### VA vs Compilation
```bash
# True VA (different artists per track)
VA-Now_100-2CD-2021-GROUP
├── 01-artist_one-song.mp3
├── 02-artist_two-track.mp3
└── 03-artist_three-hit.mp3

# Single artist compilation marketed as VA
Solution: Use actual artist name, not VA
Madonna-Greatest_Hits-CD-2021-GROUP

# DJ Mix with various tracks
VA-DJ_Name_Mix-CD-2021-GROUP
Note: Even if mixed by one DJ, tracks are by various artists
```

#### Split releases
```bash
# Two artists, equal billing
Artist_A_and_Artist_B-Split_Album-CD-2021-GROUP

# Various artists but only 2-3
Solution: If ≤3 artists, can list all
Artist_A_Artist_B_Artist_C-Collaboration-CD-2021-GROUP

# If >3 artists, use VA
VA-Collaboration-CD-2021-GROUP
```

### 5. Special Release Types

#### Promo Releases
```bash
# Radio promo
Artist-Title-PROMO-CDS-2021-GROUP

# Advance promo
Artist-Title-ADVANCE-CD-2021-GROUP

# White label promo
Artist-Title-PROMO-VINYL-2021-GROUP
Note: Requires proof photo with group name
```

#### Live Recordings Special Cases
```bash
# Multiple sources for same show
Artist-Venue-FM-2021-GROUP (FM broadcast)
Artist-Venue-SBD-2021-GROUP (Soundboard recording)
Artist-Venue-AUD-2021-GROUP (Audience recording)

# Festival recordings
Artist-Glastonbury_Festival-FM-2021-GROUP
Note: Include festival name as venue

# Radio session
Artist-BBC_Radio_1_Session-FM-2021-GROUP
Note: Include station and session type
```

#### Reissues and Special Editions
```bash
# Remastered
Artist-Album-REMASTERED-CD-2021-GROUP

# Anniversary edition
Artist-Album-25TH_ANNIVERSARY-2CD-2021-GROUP

# Deluxe edition
Artist-Album-DELUXE-3CD-2021-GROUP

# Box set
Artist-Complete_Works-BOXSET-10CD-2021-GROUP
```

### 6. Date Edge Cases

#### Unknown Year
```bash
# Completely unknown
Artist-Album-CD-XXXX-GROUP

# Decade known
Artist-Album-CD-197X-GROUP

# Approximate year
Artist-Album-CD-1975-GROUP
Note: Use best estimate if officially stated as "circa 1975"
```

#### Multiple Release Years
```bash
# Original vs reissue year
Rule: Use year of THIS specific release
Original 1970, 2021 remaster → use 2021

# Recorded vs released
Rule: Use release year, not recording year
Recorded 2019, released 2021 → use 2021
```

### 7. Group Tag Edge Cases

#### Internal Releases
```bash
# Standard internal
Artist-Album-CD-2021-GROUP_INT

# Internal web release
Artist-Album-WEB-2021-GROUP_INT

# Stream (always internal)
Artist-Album-STREAM-2021-GROUP_INT
```

### 8. Track Numbering Edge Cases

#### Multi-disc numbering
```bash
# Option 1: Continuous numbering
CD1: 01-99
CD2: 101-199
CD3: 201-299

# Option 2: Per-disc numbering
CD1: 01-20
CD2: 01-18
CD3: 01-22

# Recommendation: Use continuous for better sorting
```

#### Hidden tracks
```bash
# Listed hidden track
15-artist-hidden_track.mp3

# Unlisted bonus after silence
15-artist-last_song_plus_hidden.mp3
Note: Include in same file if part of same track
```

#### Interludes and skits
```bash
# Short interlude
07-artist-interlude.mp3

# Skit
12-artist-skit_3.mp3

# Intro/Outro
01-artist-intro.mp3
20-artist-outro.mp3
```

### 9. Featuring and Collaboration Edge Cases

#### Multiple featured artists
```bash
# Two features
01-main_artist-song_ft_artist_a_and_artist_b.mp3

# Many features (truncate if needed)
01-main_artist-song_ft_various.mp3

# Producer featuring vocalist
01-producer_ft_vocalist-song_title.mp3
```

#### Versus releases
```bash
# DJ battle
DJ_A_vs_DJ_B-Battle-VINYL-2021-GROUP

# Band vs Band
Band_A_vs_Band_B-Split-CD-2021-GROUP
```

### 10. Classical Music Edge Cases

#### Composer vs Performer
```bash
# Format: Performer as artist
London_Symphony_Orchestra-Beethoven_Symphony_No_9-CD-2021-GROUP

# Multiple composers
Various_Artists-Classical_Collection-CD-2021-GROUP
Note: Use VA if multiple composers/performers
```

#### Opus and catalog numbers
```bash
# Include in title if significant
Artist-Symphony_No_5_Op_67-CD-2021-GROUP

# Or simplify
Artist-Symphony_No_5-CD-2021-GROUP
```

### 11. Bootleg Edge Cases

#### Identifying bootlegs
```bash
Indicators:
- No label information
- Unofficial live recordings
- Studio outtakes
- Fan club only releases

Format:
Artist-Unofficial_Recording-BOOTLEG-CD-2021-GROUP
```

#### Semi-official releases
```bash
# Record Store Day exclusives (NOT bootleg)
Artist-RSD_Release-VINYL-2021-GROUP

# Label-approved live recordings (NOT bootleg)
Artist-Live_Album-CD-2021-GROUP

# Unofficial compilations (IS bootleg)
Artist-Rare_Tracks-BOOTLEG-CD-2021-GROUP
```

### 12. Soundtrack Edge Cases

#### Different types of soundtracks
```bash
# Original soundtrack (various artists)
VA-Movie_Title-OST-CD-2021-GROUP

# Score by single composer
Composer_Name-Movie_Title-SCORE-CD-2021-GROUP

# Video game soundtrack
VA-Game_Title-OST-CD-2021-GROUP

# Musical cast recording
VA-Musical_Name-CAST-CD-2021-GROUP
```

## Decision Matrix for Complex Cases

### When to use VA
```
Multiple artists? → YES
  ├─ More than 3? → YES → Use VA
  ├─ Equal billing? → NO → Use VA
  └─ Compilation? → YES → Use VA

Multiple artists? → NO
  └─ Use single artist name
```

### Source tag selection
```
Physical copy? → YES
  ├─ Pressed? → YES → Use media type (CD, VINYL, etc.)
  └─ Recordable? → YES → Use -R version (CDR, DVDR, etc.)

Physical copy? → NO
  ├─ Purchased online? → YES → WEB
  ├─ Streaming? → YES → STREAM (internal only)
  └─ Broadcast? → YES → Use broadcast type (FM, SAT, etc.)
```

### Length management priority
```
Path > 255 chars? → YES
  ├─ Remove featuring info
  ├─ Still too long? → Abbreviate title
  ├─ Still too long? → Abbreviate artist
  └─ Still too long? → Use initials

Path > 255 chars? → NO
  └─ Keep as is
```

## Testing Scenarios

### Test Case Examples
```python
# Test case 1: Unicode handling
input: "Björk - Vespertine"
expected: "Bjork-Vespertine"

# Test case 2: Length limit
input: "A" * 300 + "-Album-CD-2021-GROUP"
expected: Truncated to 255 total

# Test case 3: VA detection
input: ["Artist1 - Song1", "Artist2 - Song2"]
expected: "VA-Compilation-CD-2021-GROUP"

# Test case 4: Special characters
input: "AC/DC - Who's Next?"
expected: "AC_DC-Whos_Next"

# Test case 5: Multi-disc
input: {"discs": 3, "source": "CD"}
expected: "-3CD-"
```

## Common Pitfalls to Avoid

1. **Don't assume source** - Verify actual source
2. **Don't mix sources** - Separate different sources
3. **Don't use invalid chars** - Always sanitize
4. **Don't exceed limits** - Check length before saving
5. **Don't skip VA check** - Detect various artists
6. **Don't ignore bootlegs** - Mark appropriately
7. **Don't forget special tags** - OST, SINGLE, etc.
8. **Don't use wrong year** - Use release year
9. **Don't create custom tags** - Use standard tags only
10. **Don't forget proof** - Physical releases need proof
