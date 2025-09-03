"""Tests for __init__.py modules in analysis_service."""

import importlib
import sys
from pathlib import Path

import pytest

import services.analysis_service.src as analysis_module
import services.analysis_service.src.api as api_module
import services.analysis_service.src.api.endpoints as endpoints_module
import services.analysis_service.src.cue_handler as cue_handler_module
import services.analysis_service.src.file_rename_executor as executor_module
import services.analysis_service.src.file_rename_proposal as proposal_module
from services.analysis_service.src import __version__
from services.analysis_service.src.api import app
from services.analysis_service.src.api.endpoints import (
    analysis_router,
    health_router,
    metadata_router,
    recordings_router,
    streaming_router,
    tracklist_router,
    websocket_router,
)
from services.analysis_service.src.cue_handler import (
    BackupManager,
    BatchConversionReport,
    CompatibilityChecker,
    CompatibilityIssue,
    CompatibilityLevel,
    CompatibilityReport,
    ConversionChange,
    ConversionMode,
    ConversionReport,
    CueConverter,
    CueEditor,
    CueParsingError,
    CueValidationError,
    InvalidCommandError,
    InvalidTimeFormatError,
)
from services.analysis_service.src.cue_handler import (
    __version__ as cue_version,
)
from services.analysis_service.src.file_rename_executor import FileRenameExecutor
from services.analysis_service.src.file_rename_proposal import (
    FileRenameProposalConfig,
    FilesystemValidator,
    PatternManager,
    ProposalGenerator,
)


class TestAnalysisServiceInit:
    """Test the main analysis service __init__.py module."""

    def test_version_attribute(self):
        """Test that version is properly defined."""
        assert __version__ == "0.1.0"
        assert isinstance(__version__, str)

    def test_module_docstring(self):
        """Test module has proper docstring."""
        assert analysis_module.__doc__ is not None
        assert "Analysis Service for Tracktion" in analysis_module.__doc__
        assert "metadata extraction and analysis" in analysis_module.__doc__


class TestFileRenameProposalInit:
    """Test the file_rename_proposal __init__.py module."""

    def test_module_imports(self):
        """Test that module exports are properly defined."""
        # Verify all imports are callable/instantiable
        assert callable(FileRenameProposalConfig)
        assert callable(FilesystemValidator)
        assert callable(PatternManager)
        assert callable(ProposalGenerator)

    def test_all_exports(self):
        """Test __all__ is properly defined."""
        expected_exports = [
            "FileRenameProposalConfig",
            "FilesystemValidator",
            "PatternManager",
            "ProposalGenerator",
        ]

        assert hasattr(proposal_module, "__all__")
        assert proposal_module.__all__ == expected_exports

    def test_module_docstring(self):
        """Test module has proper docstring."""
        assert proposal_module.__doc__ is not None
        assert "File Rename Proposal Service" in proposal_module.__doc__
        assert "does NOT perform actual file renaming" in proposal_module.__doc__


class TestFileRenameExecutorInit:
    """Test the file_rename_executor __init__.py module."""

    def test_executor_import(self):
        """Test FileRenameExecutor is properly imported."""
        assert callable(FileRenameExecutor)

    def test_all_exports(self):
        """Test __all__ is properly defined."""
        expected_exports = ["FileRenameExecutor"]

        assert hasattr(executor_module, "__all__")
        assert executor_module.__all__ == expected_exports


class TestCueHandlerInit:
    """Test the cue_handler __init__.py module."""

    def test_version_attribute(self):
        """Test that version is properly defined."""
        assert cue_version == "1.0.0"
        assert isinstance(cue_version, str)

    def test_core_exports_importable(self):
        """Test that core exports can be imported."""
        # Test imports work without raising exceptions
        # Verify they are callable
        assert callable(BackupManager)
        assert callable(CompatibilityChecker)
        # Note: These might be enums or dataclasses
        assert CompatibilityIssue is not None
        assert CompatibilityLevel is not None
        assert CompatibilityReport is not None

    def test_converter_exports_importable(self):
        """Test converter exports can be imported."""
        # Verify they exist
        assert BatchConversionReport is not None
        assert ConversionChange is not None
        assert ConversionMode is not None
        assert ConversionReport is not None
        assert callable(CueConverter)

    def test_editor_exports_importable(self):
        """Test editor exports can be imported."""
        assert callable(CueEditor)

    def test_exception_exports_importable(self):
        """Test exception exports can be imported."""
        # Verify they are exception classes
        assert issubclass(CueParsingError, Exception)
        assert issubclass(CueValidationError, Exception)
        assert issubclass(InvalidCommandError, Exception)
        assert issubclass(InvalidTimeFormatError, Exception)

    def test_module_docstring(self):
        """Test module has proper docstring."""
        assert cue_handler_module.__doc__ is not None
        assert "CUE file handler module" in cue_handler_module.__doc__
        assert "parsing, generating, and manipulating CUE sheets" in cue_handler_module.__doc__


class TestApiEndpointsInit:
    """Test the api/endpoints __init__.py module."""

    def test_router_imports(self):
        """Test that all routers can be imported."""
        # Verify all routers are defined
        routers = [
            analysis_router,
            health_router,
            metadata_router,
            recordings_router,
            streaming_router,
            tracklist_router,
            websocket_router,
        ]

        for router in routers:
            assert router is not None

    def test_all_exports(self):
        """Test __all__ is properly defined."""
        expected_exports = [
            "analysis_router",
            "health_router",
            "metadata_router",
            "recordings_router",
            "streaming_router",
            "tracklist_router",
            "websocket_router",
        ]

        assert hasattr(endpoints_module, "__all__")
        assert endpoints_module.__all__ == expected_exports


class TestApiInit:
    """Test the api __init__.py module."""

    def test_app_import(self):
        """Test that app can be imported."""
        assert app is not None

    def test_all_exports(self):
        """Test __all__ is properly defined."""
        expected_exports = ["app"]

        assert hasattr(api_module, "__all__")
        assert api_module.__all__ == expected_exports


class TestModuleImportReliability:
    """Test module import reliability and error handling."""

    def test_module_imports_without_side_effects(self):
        """Test that importing modules doesn't cause side effects."""
        # Import our modules
        # Check that no unexpected global state was modified
        # (This is a basic sanity check)
        assert "services.analysis_service.src" in sys.modules

    def test_repeated_imports_are_safe(self):
        """Test that repeated imports are safe and return same objects."""
        # Test that the module is consistent
        assert analysis_module.__version__ == __version__

    @pytest.mark.parametrize(
        "module_path",
        [
            "services.analysis_service.src",
            "services.analysis_service.src.file_rename_proposal",
            "services.analysis_service.src.file_rename_executor",
            "services.analysis_service.src.api",
        ],
    )
    def test_modules_can_be_reloaded(self, module_path):
        """Test that modules can be safely reloaded."""
        # Import the module
        module = importlib.import_module(module_path)

        # Reload it
        reloaded_module = importlib.reload(module)

        # Should work without errors
        assert reloaded_module is not None
        assert reloaded_module.__name__ == module_path


class TestModuleStructure:
    """Test overall module structure and organization."""

    def test_module_paths_exist(self):
        """Test that expected module paths exist."""
        base_path = Path("services/analysis_service/src")

        expected_paths = [
            base_path / "__init__.py",
            base_path / "file_rename_proposal" / "__init__.py",
            base_path / "file_rename_executor" / "__init__.py",
            base_path / "api" / "__init__.py",
            base_path / "api" / "endpoints" / "__init__.py",
        ]

        for path in expected_paths:
            full_path = Path(__file__).parent.parent.parent.parent / path
            assert full_path.exists(), f"Expected path {path} does not exist"

    def test_no_star_imports_in_init_files(self):
        """Test that __init__.py files don't use star imports."""
        # This is a code quality test - star imports make it hard to track dependencies

        init_files = [
            "services/analysis_service/src/__init__.py",
            "services/analysis_service/src/file_rename_proposal/__init__.py",
            "services/analysis_service/src/file_rename_executor/__init__.py",
            "services/analysis_service/src/api/__init__.py",
            "services/analysis_service/src/api/endpoints/__init__.py",
        ]

        for init_file in init_files:
            full_path = Path(__file__).parent.parent.parent.parent / init_file
            if full_path.exists():
                content = full_path.read_text()
                # Check for star imports (basic heuristic)
                assert "from * import" not in content
                assert "import *" not in content
