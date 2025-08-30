# Official MP3 Release Rules 4.1
*Reference Document for Tracktion Project*
*Source: Official MP3 Release Rules 4.1 (2021-MP3COUNCIL)*

## Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://tools.ietf.org/html/rfc2119).

**Dirname**: An abbreviation for directoryname (also known as directory or folder), a file system structure in which the mp3 files are stored.

## Core Naming Rules

### 3. DIRNAME

#### 3.1 Required Format
Dirname MUST at least contain:
```
Artist Name - Title - Published Year (decimal) - Group name
```

#### 3.2 Special Tags
- All bootleg releases MUST have the word `BOOTLEG` in dirname
- Magazine CDs releases MUST be tagged as `-MAG-`
- INTERNAL releases MUST end with `_INT`
- Multi-disc releases MUST have a multi-disc source tag (`-2CD-`, `-3DVD-`, `-2TAPE-`, etc.)

#### 3.7 Various Artists
"VA" in dirname MUST be used for "various artists" as artistname.
Example: `VA-Some_Cool_Party_Mix-2CD-2021-GROUP`

#### 3.8 Original Soundtracks
To tag an OST, the `-OST-` additional tag MUST be used after the title.
Example: `VA-Some_Cool_Movie-OST-WEB-2021-GROUP`

#### 3.9 Web Singles
Releases from source `-WEB-` need an additional `-SINGLE-` tag when they contain only one track.
Exception: Single-file WEB releases that contain more than one track or are longer than 25 minutes do not require the `-SINGLE-` tag.

#### 3.10 Release Year
The MP3 release year MUST be the same as the record company and/or artist year of publication for that particular release.

#### 3.13 Tag Order
Dirname MUST have following tag order:
```
ARTIST / ALBUM_TITLE / CATALOGUE_NUMBER / ADDITIONAL_TAGS / YEAR / GROUP
```
- CATALOGUE_NUMBER and all other ADDITIONAL_TAGS are OPTIONAL
- Round brackets () around CATALOGUE_NUMBER are OPTIONAL

### 4. FILENAMES

#### 4.1 Required Format
Filename MUST at least contain:
```
Track Number - Artist Name - Song Title
```

#### 4.2 Order Requirements
- Track number MUST be first part of the filename
- Artist name MUST be before song title

#### 4.3 Various Artists
If it's an album with different artists (one artist per track), then the artist name MUST be added as well.

#### 4.5 Uniqueness
Filenames SHOULD be unique to the best of your knowledge. On a Repack or Proper, the filenames MUST be different from the original release.

### 5. DIRNAME/FILENAME RESTRICTIONS

#### 5.1 Length Limits
The length of dirname+filename MUST NOT exceed 255 characters.

#### 5.2 Character Restrictions
Files/directories MUST only contain characters:
```
a-z A-Z 0-9 _ . - ( )
```
This is to avoid problems with windows/linux-filesystem and ftp-servers.

## Source Tags

### 13.1 RETAIL SOURCES

| Source | Dir Tag Required | Dir Tag Suggested | Notes |
|--------|-----------------|-------------------|-------|
| CD SINGLE (pressed CDDA) | `-CDS-` or `-CDM-` | - | - |
| CD ALBUM (pressed CDDA) | none | `-CD-` or `-CDA-` | Multi: `-2CDA-`, `-3CDA-` |
| CD OTHER (pressed CDDA) | none | `-CD-`, `-CDEP-` | Multi: `-2CD-`, `-3CD-` |
| VINYL SINGLE | `-VLS-` or `-VINYL-` | `-VLS-` | Multi: `-2VLS-`, `-3VLS-` |
| VINYL ALBUM | `-LP-` or `-VINYL-` | `-LP-` | Multi: `-2LP-`, `-3LP-` |
| VINYL OTHER | `-VINYL-` | - | Multi: `-2VINYL-`, `-3VINYL-` |
| DVD (pressed video) | `-DVD-` | - | Multi: `-2DVD-`, `-3DVD-` |
| DVD SINGLE | `-DVDS-` or `-DVD-` | `-DVDS-` | - |
| DVDA | `-DVDA-` | - | Multi: `-2DVDA-`, `-3DVDA-` |
| SACD | `-SACD-` | - | Multi: `-2SACD-`, `-3SACD-` |
| HD DVD | `-HDDVD-` | - | Multi: `-2HDDVD-` |
| BLU-RAY DISC | `-BD-` | - | Multi: `-2BD-`, `-3BD-` |
| ANALOG TAPE | `-TAPE-` | - | Multi: `-2TAPE-`, `-3TAPE-` |
| FLASH medium | `-FLASH-` | - | SlotMusic, CompactStick |

### 13.2 DIGITAL DOWNLOADS

| Source | Dir Tag | Notes |
|--------|---------|-------|
| WEB (paid) | `-WEB-` | Audio files legally available on the net and not free |
| WEB (streaming) | `-WEB-` | From streaming services appointed by label/artist |
| WEB (private) | `-WEB-` | Digital Download Cards/Code access |

### 13.3 LIVE SOURCES

| Source | Dir Tag | Description |
|--------|---------|-------------|
| ANALOG RADIO | `-FM-` | Terrestrial radio |
| CABLE | `-CABLE-` | Analog or digital cable |
| SAT | `-SAT-` | Analog or digital satellite |
| DVB-S | `-DVBS-` | Digital video broadcast satellite |
| DVB-C | `-DVBC-` | Digital video broadcast cable |
| DVB-T | `-DVBT-` | Digital video broadcast terrestrial |
| DAB | `-DAB-` | Digital audio broadcast |
| MD | `-MD-` | MiniDisc |
| AUD | `-AUD-` | Direct live recording from audience |
| LINE | `-LINE-` | Direct recording from soundboard/mixer |
| SBD | `-SBD-` | MP3 file supplied by radio station or DJ |
| STREAM | `-STREAM-` | Webstream (must be internal) |

### 13.4 OTHER SOURCES

| Source | Dir Tag | Notes |
|--------|---------|-------|
| CDR | `-CDR-` | Small label/artist releasing on CDR |
| DVDR | `-DVDR-` | Small label/artist releasing on DVDR |
| HD DVD-R | `-HDDVDR-` | HD DVD Recordable |
| BD-R | `-BDR-` | Blu-ray Disc Recordable |
| DAT | `-DAT-` | DAT Tape |
| MP3CD | `-MP3CD-` | MP3 files on data-CD |

## Encoding Quality Standards

### 10.1 Source-Specific Bitrates

| Source | Type | Allowed Bitrates |
|--------|------|------------------|
| WEB | Music | 320 kbit CBR only |
| WEB | Audiobook | ANY CBR bitrate or VBR |
| MP3CD | Music | 192-320 kbit CBR (highest available) or VBR |
| MP3CD | Audiobook | ANY CBR bitrate or VBR |
| FLASH | All | 192-320 kbit CBR (highest available) or VBR |

### 10.2 Encoding Standards

#### 10.2.1 Lossless WEB Sources
- MUST use LAME 3.100
- Preset: `-b 320`
- No additional switches affecting quality

#### 10.2.2 All Other Sources (non-WEB)
- MUST use LAME 3.100
- Preset: `-V0`
- No additional switches affecting quality

### 10.3 Sampling Rate
- MUST be either 44.1khz or 48khz
- 32khz is NOT allowed
- Downsampling to 44.1 or 48khz is allowed (if source is better)
- Upsampling is NOT allowed

## ID3 Tags

### 15.1 Required Tags
All mp3 files need:
- ID3 v1.1 tag
- ID3 v2 tag

Must include:
- Artist
- Title
- Album

Should include (but not nukeable if incorrect):
- Year
- Genre
- Tracknumber

### 15.2 Picture Size Limit
Maximum allowed size for pictures in ID3 tags: 2MB (2,097,152 bytes) per mp3 file

## Examples

### Standard Releases
```
Artist_Name-Album_Title-CD-2021-GROUP/
├── 00-artist_name-album_title-cd-2021-group.nfo
├── 00-artist_name-album_title-cd-2021-group.sfv
├── 00-artist_name-album_title-cd-2021-proof.jpg
├── 01-artist_name-song_title.mp3
├── 02-artist_name-another_song.mp3
└── 03-artist_name-third_track.mp3
```

### Various Artists
```
VA-Compilation_Name-2CD-2021-GROUP/
├── CD1/
│   ├── 101-first_artist-track_one.mp3
│   ├── 102-second_artist-track_two.mp3
│   └── 103-third_artist-track_three.mp3
└── CD2/
    ├── 201-fourth_artist-track_one.mp3
    ├── 202-fifth_artist-track_two.mp3
    └── 203-sixth_artist-track_three.mp3
```

### Web Single
```
Artist_Name-Track_Title-SINGLE-WEB-2021-GROUP/
├── 00-artist_name-track_title-single-web-2021-group.nfo
├── 00-artist_name-track_title-single-web-2021-group.sfv
└── 01-artist_name-track_title.mp3
```

### Live Recording
```
Artist_Name-Live_at_Venue-FM-2021-GROUP/
├── 00-artist_name-live_at_venue-fm-2021-group.nfo
├── 00-artist_name-live_at_venue-fm-2021-group.sfv
├── 01-artist_name-intro_live.mp3
├── 02-artist_name-song_one_live.mp3
└── 03-artist_name-song_two_live.mp3
```

## Implementation Notes for Tracktion

### Priority Implementation Areas

1. **Core Naming Engine**
   - Dirname structure generation
   - Filename formatting
   - Character restriction enforcement
   - Length limit handling

2. **Source Detection**
   - Automatic source type identification
   - Proper tag application
   - Multi-disc notation

3. **Special Cases**
   - VA (Various Artists) handling
   - OST tagging
   - SINGLE tag for web releases
   - Bootleg detection

4. **Character Handling**
   - Replace invalid characters
   - Handle foreign characters
   - Maintain filesystem compatibility
   - Smart truncation for length limits

5. **Integration Points**
   - `renaming_service` updates
   - `media_discovery_service` compatibility
   - File organization patterns
   - Metadata extraction alignment

### Validation Checklist
- [ ] Dirname contains all required elements
- [ ] Filename starts with track number
- [ ] Only valid characters used
- [ ] Total path length ≤ 255 characters
- [ ] Source tags properly applied
- [ ] VA releases correctly formatted
- [ ] Multi-disc notation present when needed
- [ ] ID3 tags populated correctly
