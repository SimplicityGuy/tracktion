# Data Models (Refined and Finalized)

[cite\_start]This section defines the core data models and their relationships, which will be the basis for the database schema[cite: 2623]. These models are designed to be extensible and robust, directly addressing the project's functional requirements.

### **Recording**

**Purpose:** Represents a single digital music file, such as a live set or concert recording. It is the central entity that other data models will relate to.

**Key Attributes:**

  * `id`: Unique identifier (UUID).
  * `file_path`: String - The full path to the file on the file system.
  * `file_name`: String - The standardized name of the file.
  * `created_at`: Datetime - Timestamp when the file was first cataloged.

**Relationships:**

  * `HAS_METADATA`: A `Recording` has many `Metadata` entries.
  * `HAS_TRACKLIST`: A `Recording` has one `Tracklist`.

### **Metadata**

**Purpose:** Stores extensible, key-value data about a recording. This model addresses the need to easily add new characteristics (e.g., mood, genre) without altering the database schema.

**Key Attributes:**

  * `id`: Unique identifier (UUID).
  * `recording_id`: Foreign key linking to the `Recording` model.
  * `key`: String - The name of the metadata (e.g., "bpm", "mood").
  * `value`: String - The value of the metadata (e.g., "128", "energetic").

### **Tracklist**

**Purpose:** Represents the tracklist for a specific recording, detailing the songs played and their timing.

**Key Attributes:**

  * `id`: Unique identifier (UUID).
  * `recording_id`: Foreign key linking to the `Recording` model.
  * `source`: String - The source of the tracklist (e.g., "manual", "`1001tracklists.com`").
  * `tracks`: JSONB/Array of objects - An array of tracks, each containing a title, artist, and start time.
  * `cue_file_path`: String - The path to the generated `.cue` file.

**Relationships:**

  * `BELONGS_TO`: A `Tracklist` belongs to a single `Recording`.
