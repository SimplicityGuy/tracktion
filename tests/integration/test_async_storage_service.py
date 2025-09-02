"""
Integration tests for Storage service S3 operations with mocked AWS S3.

Tests upload/download operations, error handling, and S3 integration
using moto for AWS S3 mocking.
"""

import io
import json
import logging
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import boto3
import pytest
from moto import mock_s3

from services.analysis_service.src.async_storage_handler import AsyncStorageHandler

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_BUCKET = "tracktion-test-bucket"
AWS_REGION = "us-east-1"
POSTGRES_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://tracktion:tracktion_password@localhost:5432/tracktion_test"
)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
REDIS_URL = "redis://localhost:6379/1"


class MockS3StorageService:
    """Mock S3 storage service for testing."""

    def __init__(self, bucket_name: str = TEST_BUCKET):
        self.bucket_name = bucket_name
        self.s3_client = None

    def _ensure_s3_client(self):
        """Ensure S3 client is initialized."""
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3",
                region_name=AWS_REGION,
                aws_access_key_id="testing",
                aws_secret_access_key="testing",
            )

    def upload_file(self, file_path: str, s3_key: str) -> tuple[bool, str, str | None]:
        """
        Upload file to S3.

        Returns:
            Tuple of (success, s3_url, error_message)
        """
        try:
            self._ensure_s3_client()

            with Path(file_path).open("rb") as file_data:
                self.s3_client.upload_fileobj(
                    file_data, self.bucket_name, s3_key, ExtraArgs={"ContentType": self._get_content_type(file_path)}
                )

            s3_url = f"https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
            return True, s3_url, None

        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False, "", str(e)

    def upload_content(
        self, content: str | bytes, s3_key: str, content_type: str = "text/plain"
    ) -> tuple[bool, str, str | None]:
        """
        Upload content directly to S3.

        Returns:
            Tuple of (success, s3_url, error_message)
        """
        try:
            self._ensure_s3_client()

            if isinstance(content, str):
                content = content.encode("utf-8")

            content_stream = io.BytesIO(content)

            self.s3_client.upload_fileobj(
                content_stream, self.bucket_name, s3_key, ExtraArgs={"ContentType": content_type}
            )

            s3_url = f"https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
            return True, s3_url, None

        except Exception as e:
            logger.error(f"Failed to upload content to S3: {e}")
            return False, "", str(e)

    def download_file(self, s3_key: str, local_path: str) -> tuple[bool, str | None]:
        """
        Download file from S3.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            self._ensure_s3_client()
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True, None

        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False, str(e)

    def download_content(self, s3_key: str) -> tuple[bool, str | bytes | None, str | None]:
        """
        Download content directly from S3.

        Returns:
            Tuple of (success, content, error_message)
        """
        try:
            self._ensure_s3_client()
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response["Body"].read()

            # Try to decode as UTF-8, otherwise return bytes
            try:
                return True, content.decode("utf-8"), None
            except UnicodeDecodeError:
                return True, content, None

        except Exception as e:
            logger.error(f"Failed to download content from S3: {e}")
            return False, None, str(e)

    def delete_file(self, s3_key: str) -> tuple[bool, str | None]:
        """
        Delete file from S3.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            self._ensure_s3_client()
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True, None

        except Exception as e:
            logger.error(f"Failed to delete file from S3: {e}")
            return False, str(e)

    def list_files(self, prefix: str = "") -> tuple[bool, list[str], str | None]:
        """
        List files in S3 bucket with optional prefix.

        Returns:
            Tuple of (success, file_keys, error_message)
        """
        try:
            self._ensure_s3_client()

            kwargs = {"Bucket": self.bucket_name}
            if prefix:
                kwargs["Prefix"] = prefix

            response = self.s3_client.list_objects_v2(**kwargs)

            if "Contents" not in response:
                return True, [], None

            file_keys = [obj["Key"] for obj in response["Contents"]]
            return True, file_keys, None

        except Exception as e:
            logger.error(f"Failed to list files in S3: {e}")
            return False, [], str(e)

    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self._ensure_s3_client()
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except Exception:
            return False

    def get_file_info(self, s3_key: str) -> tuple[bool, dict[str, Any] | None, str | None]:
        """
        Get file metadata from S3.

        Returns:
            Tuple of (success, metadata, error_message)
        """
        try:
            self._ensure_s3_client()
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)

            metadata = {
                "size": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
                "content_type": response.get("ContentType"),
                "etag": response.get("ETag", "").strip('"'),
                "metadata": response.get("Metadata", {}),
            }

            return True, metadata, None

        except Exception as e:
            logger.error(f"Failed to get file info from S3: {e}")
            return False, None, str(e)

    def store_cue_file(self, file_path: str, content: str) -> tuple[bool, str, str | None]:
        """
        Store CUE file content (compatibility method).

        Returns:
            Tuple of (success, stored_path, error_message)
        """
        return self.upload_content(content, file_path, "text/plain")

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type based on file extension."""
        extension = Path(file_path).suffix.lower()
        content_types = {
            ".txt": "text/plain",
            ".json": "application/json",
            ".xml": "application/xml",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".cue": "text/plain",
        }
        return content_types.get(extension, "application/octet-stream")


@pytest.fixture
def s3_mock():
    """Create mock S3 environment."""
    with mock_s3():
        # Create S3 client and bucket
        s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )

        s3_client.create_bucket(Bucket=TEST_BUCKET)
        yield s3_client


@pytest.fixture
def storage_service(s3_mock):
    """Create storage service instance with mocked S3."""
    return MockS3StorageService(TEST_BUCKET)


@pytest.fixture
def temp_file():
    """Create temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        test_content = "This is a test file content for S3 upload testing."
        f.write(test_content)
        temp_path = f.name

    yield temp_path, test_content

    # Cleanup
    temp_file = Path(temp_path)
    if temp_file.exists():
        temp_file.unlink()


class TestMockS3StorageServiceBasicOperations:
    """Test basic S3 storage operations."""

    def test_upload_and_download_file(self, storage_service: MockS3StorageService, temp_file):
        """Test file upload and download operations."""
        temp_path, original_content = temp_file
        s3_key = f"test/files/{uuid4()}.txt"

        # Upload file
        success, s3_url, error = storage_service.upload_file(temp_path, s3_key)
        assert success is True
        assert s3_url.startswith(f"https://{TEST_BUCKET}.s3.")
        assert s3_url.endswith(s3_key)
        assert error is None

        # Verify file exists
        assert storage_service.file_exists(s3_key) is True

        # Download file to new location
        with tempfile.NamedTemporaryFile(delete=False) as download_file:
            download_path = download_file.name

        try:
            success, error = storage_service.download_file(s3_key, download_path)
            assert success is True
            assert error is None

            # Verify content
            with Path(download_path).open() as f:
                downloaded_content = f.read()
            assert downloaded_content == original_content

        finally:
            download_file = Path(download_path)
            if download_file.exists():
                download_file.unlink()

    def test_upload_and_download_content(self, storage_service: MockS3StorageService):
        """Test direct content upload and download."""
        test_content = "Direct content upload test with Unicode: 🚀🌟💫"
        s3_key = f"test/content/{uuid4()}.txt"

        # Upload content
        success, s3_url, error = storage_service.upload_content(
            test_content, s3_key, content_type="text/plain; charset=utf-8"
        )
        assert success is True
        assert s3_url.startswith(f"https://{TEST_BUCKET}.s3.")
        assert error is None

        # Download content
        success, downloaded_content, error = storage_service.download_content(s3_key)
        assert success is True
        assert downloaded_content == test_content
        assert error is None

    def test_upload_json_content(self, storage_service: MockS3StorageService):
        """Test uploading JSON content."""
        test_data = {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {"type": "test_data", "version": "1.0.0"},
            "values": [1, 2, 3, 4, 5],
        }

        json_content = json.dumps(test_data, indent=2)
        s3_key = f"test/json/{uuid4()}.json"

        # Upload JSON
        success, s3_url, error = storage_service.upload_content(json_content, s3_key, content_type="application/json")
        assert success is True
        assert error is None

        # Download and parse
        success, downloaded_content, error = storage_service.download_content(s3_key)
        assert success is True
        assert error is None

        parsed_data = json.loads(downloaded_content)
        assert parsed_data == test_data
        assert parsed_data["id"] == test_data["id"]
        assert parsed_data["values"] == [1, 2, 3, 4, 5]

    def test_binary_file_upload(self, storage_service: MockS3StorageService):
        """Test binary file upload and download."""
        # Create binary test data
        binary_data = bytes(range(256))  # 256 bytes of binary data
        s3_key = f"test/binary/{uuid4()}.bin"

        # Upload binary content
        success, s3_url, error = storage_service.upload_content(
            binary_data, s3_key, content_type="application/octet-stream"
        )
        assert success is True
        assert error is None

        # Download binary content
        success, downloaded_content, error = storage_service.download_content(s3_key)
        assert success is True
        assert isinstance(downloaded_content, bytes)
        assert downloaded_content == binary_data
        assert error is None

    def test_file_operations(self, storage_service: MockS3StorageService):
        """Test file management operations."""
        test_content = "File operations test"
        s3_key = f"test/operations/{uuid4()}.txt"

        # Initially file should not exist
        assert storage_service.file_exists(s3_key) is False

        # Upload file
        success, _, error = storage_service.upload_content(test_content, s3_key)
        assert success is True
        assert error is None

        # Now file should exist
        assert storage_service.file_exists(s3_key) is True

        # Get file info
        success, file_info, error = storage_service.get_file_info(s3_key)
        assert success is True
        assert error is None
        assert file_info is not None
        assert file_info["size"] == len(test_content.encode("utf-8"))
        assert file_info["content_type"] == "text/plain"
        assert "last_modified" in file_info
        assert "etag" in file_info

        # Delete file
        success, error = storage_service.delete_file(s3_key)
        assert success is True
        assert error is None

        # File should no longer exist
        assert storage_service.file_exists(s3_key) is False

    def test_list_files_operation(self, storage_service: MockS3StorageService):
        """Test file listing with prefixes."""
        # Create test files with different prefixes
        test_files = [
            ("test/listing/file1.txt", "Content 1"),
            ("test/listing/file2.txt", "Content 2"),
            ("test/listing/subdir/file3.txt", "Content 3"),
            ("other/path/file4.txt", "Content 4"),
        ]

        # Upload test files
        for s3_key, content in test_files:
            success, _, error = storage_service.upload_content(content, s3_key)
            assert success is True
            assert error is None

        # List all files
        success, all_files, error = storage_service.list_files()
        assert success is True
        assert error is None
        assert len(all_files) >= len(test_files)

        for s3_key, _ in test_files:
            assert s3_key in all_files

        # List files with prefix
        success, listing_files, error = storage_service.list_files("test/listing/")
        assert success is True
        assert error is None
        assert len(listing_files) == 3  # file1.txt, file2.txt, subdir/file3.txt

        expected_listing_files = ["test/listing/file1.txt", "test/listing/file2.txt", "test/listing/subdir/file3.txt"]
        for expected_file in expected_listing_files:
            assert expected_file in listing_files

        # List files with more specific prefix
        success, subdir_files, error = storage_service.list_files("test/listing/subdir/")
        assert success is True
        assert error is None
        assert len(subdir_files) == 1
        assert "test/listing/subdir/file3.txt" in subdir_files


class TestMockS3StorageServiceErrorHandling:
    """Test error handling scenarios."""

    def test_upload_nonexistent_file(self, storage_service: MockS3StorageService):
        """Test uploading a non-existent file."""
        nonexistent_path = "/path/that/does/not/exist.txt"
        s3_key = f"test/error/{uuid4()}.txt"

        success, s3_url, error = storage_service.upload_file(nonexistent_path, s3_key)
        assert success is False
        assert s3_url == ""
        assert error is not None
        assert "No such file or directory" in error or "cannot find" in error.lower()

    def test_download_nonexistent_file(self, storage_service: MockS3StorageService):
        """Test downloading a non-existent file."""
        nonexistent_key = f"test/nonexistent/{uuid4()}.txt"

        with tempfile.NamedTemporaryFile() as temp_file:
            success, error = storage_service.download_file(nonexistent_key, temp_file.name)
            assert success is False
            assert error is not None

        # Test content download
        success, content, error = storage_service.download_content(nonexistent_key)
        assert success is False
        assert content is None
        assert error is not None

    def test_get_info_nonexistent_file(self, storage_service: MockS3StorageService):
        """Test getting info for non-existent file."""
        nonexistent_key = f"test/nonexistent/{uuid4()}.txt"

        success, file_info, error = storage_service.get_file_info(nonexistent_key)
        assert success is False
        assert file_info is None
        assert error is not None

    def test_delete_nonexistent_file(self, storage_service: MockS3StorageService):
        """Test deleting a non-existent file (should succeed in S3)."""
        nonexistent_key = f"test/nonexistent/{uuid4()}.txt"

        # S3 delete operations typically succeed even if file doesn't exist
        success, error = storage_service.delete_file(nonexistent_key)
        assert success is True  # S3 behavior
        assert error is None


class TestMockS3StorageServiceCUEFileOperations:
    """Test CUE file specific operations."""

    def test_store_cue_file(self, storage_service: MockS3StorageService):
        """Test storing CUE file content."""
        cue_content = """REM GENERATED BY Tracklist Service
REM FORMAT STANDARD
REM DATE 2024-01-01
TITLE "Test Mix"
PERFORMER "Test Artist"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Artist 1"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track 2"
    PERFORMER "Artist 2"
    INDEX 01 03:45:32
"""

        cue_path = f"cue_files/{uuid4()}/standard.cue"

        # Store CUE file
        success, stored_path, error = storage_service.store_cue_file(cue_path, cue_content)
        assert success is True
        assert error is None

        # Verify stored content
        success, downloaded_content, error = storage_service.download_content(cue_path)
        assert success is True
        assert downloaded_content == cue_content
        assert error is None

        # Verify it's stored as text/plain
        success, file_info, error = storage_service.get_file_info(cue_path)
        assert success is True
        assert file_info["content_type"] == "text/plain"

    def test_multiple_cue_formats(self, storage_service: MockS3StorageService):
        """Test storing multiple CUE formats for the same tracklist."""
        tracklist_id = str(uuid4())
        formats = ["standard", "cdj", "traktor", "serato", "rekordbox"]

        cue_files = {}
        for format_name in formats:
            cue_content = f"""REM GENERATED BY Tracklist Service
REM FORMAT {format_name.upper()}
TITLE "Test Mix - {format_name}"
PERFORMER "Test Artist"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    INDEX 01 00:00:00
"""
            cue_path = f"cue_files/{tracklist_id}/{format_name}.cue"
            cue_files[format_name] = (cue_path, cue_content)

            # Store CUE file
            success, _, error = storage_service.store_cue_file(cue_path, cue_content)
            assert success is True
            assert error is None

        # Verify all formats are stored
        for format_name, (cue_path, original_content) in cue_files.items():
            success, downloaded_content, error = storage_service.download_content(cue_path)
            assert success is True
            assert downloaded_content == original_content
            assert f"FORMAT {format_name.upper()}" in downloaded_content

        # List all CUE files for this tracklist
        success, cue_file_keys, error = storage_service.list_files(f"cue_files/{tracklist_id}/")
        assert success is True
        assert len(cue_file_keys) == len(formats)

        for format_name in formats:
            expected_key = f"cue_files/{tracklist_id}/{format_name}.cue"
            assert expected_key in cue_file_keys


class TestMockS3StorageServicePerformance:
    """Test performance characteristics."""

    def test_bulk_upload_performance(self, storage_service: MockS3StorageService):
        """Test bulk upload operations."""

        num_files = 50
        file_size = 1024  # 1KB per file
        test_content = "x" * file_size

        # Upload multiple files
        start_time = time.time()
        uploaded_files = []

        for i in range(num_files):
            s3_key = f"test/performance/bulk_{i}.txt"
            success, s3_url, error = storage_service.upload_content(test_content, s3_key)
            assert success is True
            assert error is None
            uploaded_files.append(s3_key)

        upload_duration = time.time() - start_time

        # Download all files
        start_time = time.time()
        download_count = 0

        for s3_key in uploaded_files:
            success, content, error = storage_service.download_content(s3_key)
            if success and content == test_content:
                download_count += 1

        download_duration = time.time() - start_time

        # Performance assertions
        assert download_count == num_files
        assert upload_duration < 10.0  # Should upload 50 files in under 10 seconds
        assert download_duration < 5.0  # Should download 50 files in under 5 seconds

        # Calculate throughput
        upload_throughput = num_files / upload_duration
        download_throughput = num_files / download_duration

        logger.info(f"Upload throughput: {upload_throughput:.1f} files/sec")
        logger.info(f"Download throughput: {download_throughput:.1f} files/sec")

        # Should achieve reasonable throughput with mocked S3
        assert upload_throughput > 5  # At least 5 uploads per second
        assert download_throughput > 10  # At least 10 downloads per second

    def test_large_file_handling(self, storage_service: MockS3StorageService):
        """Test handling of large files."""

        # Create 1MB file content
        large_content = "Large file content: " + ("x" * (1024 * 1024 - 20))
        s3_key = f"test/performance/large_file_{uuid4()}.txt"

        # Upload large file
        start_time = time.time()
        success, s3_url, error = storage_service.upload_content(large_content, s3_key)
        upload_duration = time.time() - start_time

        assert success is True
        assert error is None

        # Download large file
        start_time = time.time()
        success, downloaded_content, error = storage_service.download_content(s3_key)
        download_duration = time.time() - start_time

        assert success is True
        assert downloaded_content == large_content
        assert len(downloaded_content) == len(large_content)
        assert error is None

        # Performance should be reasonable even for large files
        assert upload_duration < 5.0  # Upload 1MB in under 5 seconds
        assert download_duration < 3.0  # Download 1MB in under 3 seconds

        logger.info(f"Large file upload: {upload_duration:.3f}s")
        logger.info(f"Large file download: {download_duration:.3f}s")

        # Verify file info
        success, file_info, error = storage_service.get_file_info(s3_key)
        assert success is True
        assert file_info["size"] == len(large_content.encode("utf-8"))


class TestAsyncStorageIntegration:
    """Test integration with AsyncStorageHandler."""

    @pytest.fixture
    async def storage_handler(self, s3_mock):
        """Create AsyncStorageHandler with mocked services."""
        # Note: This is a simplified version since full async integration
        # would require running database and Redis services
        handler = AsyncStorageHandler(
            postgres_url=POSTGRES_URL, neo4j_uri=NEO4J_URI, neo4j_auth=NEO4J_AUTH, redis_url=REDIS_URL
        )

        # Mock the storage components for testing
        handler._mock_storage_service = MockS3StorageService(TEST_BUCKET)

        yield handler

        await handler.close()

    @pytest.mark.asyncio
    async def test_storage_handler_initialization(self, storage_handler):
        """Test storage handler initialization."""
        assert storage_handler is not None
        assert hasattr(storage_handler, "_mock_storage_service")

        # Test that components are initialized
        assert storage_handler.db_manager is not None
        assert storage_handler.recording_repo is not None
        assert storage_handler.metadata_repo is not None
        assert storage_handler.tracklist_repo is not None

    def test_storage_service_integration(self, storage_service: MockS3StorageService):
        """Test storage service integration with actual workflow."""
        # Simulate storing analysis results with files
        recording_id = str(uuid4())
        analysis_type = "spectrum_analysis"

        # Create analysis data with file attachments
        spectrum_data = {
            "frequencies": list(range(0, 22050, 100)),  # 0-22kHz in 100Hz steps
            "magnitudes": [0.5 + (i % 10) * 0.05 for i in range(221)],
            "metadata": {"sample_rate": 44100, "fft_size": 2048, "window": "hanning"},
        }

        # Store spectrum data as JSON
        spectrum_key = f"analysis/{recording_id}/spectrum/{analysis_type}.json"
        spectrum_json = json.dumps(spectrum_data, indent=2)

        success, s3_url, error = storage_service.upload_content(spectrum_json, spectrum_key, "application/json")
        assert success is True
        assert error is None

        # Store additional binary analysis data
        binary_data = bytes([i % 256 for i in range(1024)])  # 1KB of binary data
        binary_key = f"analysis/{recording_id}/spectrum/raw_data.bin"

        success, binary_url, error = storage_service.upload_content(binary_data, binary_key, "application/octet-stream")
        assert success is True
        assert error is None

        # Verify both files are stored
        assert storage_service.file_exists(spectrum_key) is True
        assert storage_service.file_exists(binary_key) is True

        # List all analysis files for this recording
        success, analysis_files, error = storage_service.list_files(f"analysis/{recording_id}/")
        assert success is True
        assert len(analysis_files) == 2
        assert spectrum_key in analysis_files
        assert binary_key in analysis_files

        # Verify content integrity
        success, retrieved_json, error = storage_service.download_content(spectrum_key)
        assert success is True
        retrieved_data = json.loads(retrieved_json)
        assert retrieved_data == spectrum_data

        success, retrieved_binary, error = storage_service.download_content(binary_key)
        assert success is True
        assert retrieved_binary == binary_data


@pytest.mark.integration
@pytest.mark.requires_docker
class TestS3StorageServiceIntegration:
    """Integration tests that would work with real AWS S3 (when configured)."""

    def test_complete_storage_workflow(self, storage_service: MockS3StorageService):
        """Test complete storage workflow from creation to cleanup."""
        project_id = str(uuid4())

        # Step 1: Store project metadata
        project_metadata = {
            "id": project_id,
            "name": "Test Project",
            "created_at": datetime.now(UTC).isoformat(),
            "type": "audio_analysis",
            "settings": {"sample_rate": 44100, "bit_depth": 16, "channels": 2},
        }

        metadata_key = f"projects/{project_id}/metadata.json"
        success, _, error = storage_service.upload_content(
            json.dumps(project_metadata), metadata_key, "application/json"
        )
        assert success is True
        assert error is None

        # Step 2: Store analysis results
        analysis_results = []
        analysis_types = ["bpm", "key", "mood", "genre"]

        for analysis_type in analysis_types:
            result_data = {
                "type": analysis_type,
                "value": f"test_{analysis_type}_result",
                "confidence": 0.85 + (len(analysis_results) * 0.03),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            result_key = f"projects/{project_id}/analysis/{analysis_type}.json"
            success, _, error = storage_service.upload_content(json.dumps(result_data), result_key, "application/json")
            assert success is True
            assert error is None

            analysis_results.append((result_key, result_data))

        # Step 3: Store generated CUE files
        cue_formats = ["standard", "traktor", "serato"]
        cue_files = []

        for cue_format in cue_formats:
            cue_content = f"""REM PROJECT {project_id}
REM FORMAT {cue_format.upper()}
TITLE "Generated Mix"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Test Track"
    INDEX 01 00:00:00
"""

            cue_key = f"projects/{project_id}/cue/{cue_format}.cue"
            success, _, error = storage_service.store_cue_file(cue_key, cue_content)
            assert success is True
            assert error is None

            cue_files.append((cue_key, cue_content))

        # Step 4: Verify all files exist and content is correct
        success, all_files, error = storage_service.list_files(f"projects/{project_id}/")
        assert success is True
        assert error is None

        expected_file_count = 1 + len(analysis_results) + len(cue_files)  # metadata + analysis + cue
        assert len(all_files) == expected_file_count

        # Verify metadata
        success, metadata_content, error = storage_service.download_content(metadata_key)
        assert success is True
        retrieved_metadata = json.loads(metadata_content)
        assert retrieved_metadata["id"] == project_id
        assert retrieved_metadata["name"] == "Test Project"

        # Verify analysis results
        for result_key, original_data in analysis_results:
            success, result_content, error = storage_service.download_content(result_key)
            assert success is True
            retrieved_result = json.loads(result_content)
            assert retrieved_result == original_data

        # Verify CUE files
        for cue_key, original_content in cue_files:
            success, cue_content, error = storage_service.download_content(cue_key)
            assert success is True
            assert cue_content == original_content
            assert f"PROJECT {project_id}" in cue_content

        # Step 5: Cleanup - delete all project files
        deleted_files = []
        for file_key in all_files:
            success, error = storage_service.delete_file(file_key)
            assert success is True
            assert error is None
            deleted_files.append(file_key)

        # Verify cleanup
        for file_key in deleted_files:
            assert storage_service.file_exists(file_key) is False

        # Verify project directory is empty
        success, remaining_files, error = storage_service.list_files(f"projects/{project_id}/")
        assert success is True
        assert len(remaining_files) == 0
