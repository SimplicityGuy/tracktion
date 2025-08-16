# Requirements
### **Functional**
1.  **FR1:** The system shall automatically scan a user-defined directory for new audio files.
2.  **FR2:** The system shall catalog newly detected audio files in a PostgreSQL database.
3.  **FR3:** The system shall analyze audio files to extract metadata such as BPM, musical key, and mood, storing this in a Neo4j graph database.
4.  **FR4:** The system shall query `1001tracklists.com` to find tracklists for recordings that do not have them.
5.  **FR5:** The system shall automatically generate a `.cue` file from retrieved or existing tracklist data.
6.  **FR6:** The system shall rename audio files based on a user-defined pattern.

### **Non Functional**
1.  **NFR1:** The system must be deployable via Docker containers.
2.  **NFR2:** The system shall be scalable to handle a large and growing collection of recordings.
3.  **NFR3:** The system shall use a message queue (e.g., RabbitMQ) for asynchronous processing to ensure decoupling and reliability.
4.  **NFR4:** All application configuration, including directory paths and datastore connections, must be managed via YAML files.
