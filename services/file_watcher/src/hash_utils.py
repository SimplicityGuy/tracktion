"""Hash utility functions for file processing."""

from enum import Enum


class HashType(Enum):
    """Enumeration of hash types and their use cases."""

    SHA256 = "sha256"
    XXH128 = "xxh128"


class HashStrategy:
    """Strategy for selecting which hash to use based on context."""

    @staticmethod
    def for_duplicate_detection() -> HashType:
        """Get hash type for duplicate detection.

        XXH128 is significantly faster for duplicate detection.

        Returns:
            HashType.XXH128 for fast duplicate detection
        """
        return HashType.XXH128

    @staticmethod
    def for_integrity_verification() -> HashType:
        """Get hash type for data integrity verification.

        SHA256 provides cryptographic integrity guarantees.

        Returns:
            HashType.SHA256 for cryptographic integrity
        """
        return HashType.SHA256

    @staticmethod
    def for_database_lookup() -> HashType:
        """Get hash type for database lookups.

        XXH128 is preferred for performance reasons.

        Returns:
            HashType.XXH128 for fast database lookups
        """
        return HashType.XXH128

    @staticmethod
    def for_api_response() -> list[HashType]:
        """Get hash types to include in API responses.

        Both hashes provide flexibility for clients.

        Returns:
            List containing both hash types
        """
        return [HashType.SHA256, HashType.XXH128]

    @staticmethod
    def select_hash(sha256_hash: str | None, xxh128_hash: str | None, purpose: HashType) -> str | None:
        """Select the appropriate hash based on purpose.

        Args:
            sha256_hash: SHA256 hash value
            xxh128_hash: XXH128 hash value
            purpose: The HashType indicating which hash to use

        Returns:
            The selected hash value, or None if not available
        """
        if purpose == HashType.SHA256:
            return sha256_hash
        elif purpose == HashType.XXH128:
            return xxh128_hash
        else:
            return None
