"""Tests for the file rename proposal engine."""

import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from services.file_rename_service.app.proposal.batch_processor import (
    BatchProcessingResult,
    BatchProcessor,
)
from services.file_rename_service.app.proposal.cache import ProposalCache
from services.file_rename_service.app.proposal.conflicts import (
    check_duplicate,
    detect_conflicts,
    generate_unique_name,
    resolve_conflict,
    validate_filename,
)
from services.file_rename_service.app.proposal.explainer import RenameExplainer
from services.file_rename_service.app.proposal.generator import ProposalGenerator
from services.file_rename_service.app.proposal.models import (
    ConflictResolution,
    NamingTemplate,
    RenameProposal,
)
from services.file_rename_service.app.proposal.scorer import ConfidenceScorer
from services.file_rename_service.app.proposal.templates import TemplateManager


class TestProposalModels:
    """Test data models for rename proposals."""

    def test_rename_proposal_creation(self) -> None:
        """Test creating a RenameProposal."""
        proposal = RenameProposal(
            original_filename="test.mp3",
            proposed_filename="Artist - Song.mp3",
            confidence_score=0.85,
            explanation="Based on metadata analysis",
            patterns_used=["artist_title"],
            alternatives=["Song - Artist.mp3"],
            conflict_status="none",
        )
        assert proposal.original_filename == "test.mp3"
        assert proposal.confidence_score == 0.85
        assert len(proposal.alternatives) == 1

    def test_rename_proposal_validation(self) -> None:
        """Test RenameProposal validation."""
        # Test invalid confidence score
        with pytest.raises(ValueError):
            RenameProposal(
                original_filename="test.mp3",
                proposed_filename="new.mp3",
                confidence_score=1.5,  # Invalid: > 1.0
                explanation="test",
                patterns_used=[],
                alternatives=[],
                conflict_status="none",
            )

        # Test invalid conflict status
        with pytest.raises(ValueError):
            RenameProposal(
                original_filename="test.mp3",
                proposed_filename="new.mp3",
                confidence_score=0.5,
                explanation="test",
                patterns_used=[],
                alternatives=[],
                conflict_status="invalid",  # Invalid status
            )

    def test_naming_template_creation(self) -> None:
        """Test creating a NamingTemplate."""
        template = NamingTemplate(
            id="template1",
            name="Concert Template",
            pattern="{artist} - {date} - {venue}",
            user_id="user123",
            created_at=datetime.now(UTC),
            usage_count=5,
        )
        assert template.name == "Concert Template"
        assert template.usage_count == 5
        assert template.is_active is True  # Default value

    def test_conflict_resolution_creation(self) -> None:
        """Test creating a ConflictResolution."""
        resolution = ConflictResolution(
            strategy="append_number",
            existing_file="existing.mp3",
            proposed_action="Rename to existing (1).mp3",
        )
        assert resolution.strategy == "append_number"


class TestProposalGenerator:
    """Test the proposal generator."""

    @pytest.mark.asyncio
    async def test_generate_proposal_basic(self) -> None:
        """Test basic proposal generation."""
        generator = ProposalGenerator()

        # Mock ML predictor
        with patch.object(generator.predictor, "predict") as mock_predict:
            mock_predict.return_value = {
                "predictions": [
                    {"suggested_name": "Artist - Song.mp3", "confidence": 0.8},
                    {"suggested_name": "Song - Artist.mp3", "confidence": 0.6},
                    {"suggested_name": "Artist_Song.mp3", "confidence": 0.5},
                ],
                "confidence": 0.8,
            }

            proposal = await generator.generate_proposal("test.mp3")

            assert proposal.original_filename == "test.mp3"
            assert proposal.confidence_score > 0
            assert len(proposal.alternatives) >= 3

    @pytest.mark.asyncio
    async def test_generate_proposal_with_template(self) -> None:
        """Test proposal generation with custom template."""
        generator = ProposalGenerator()

        template = NamingTemplate(
            id="t1",
            name="Test",
            pattern="{artist} - {title}",
            user_id="user1",
            created_at=datetime.now(UTC),
            usage_count=0,
        )

        with patch.object(generator.predictor, "predict") as mock_predict:
            mock_predict.return_value = {
                "predictions": [
                    {"suggested_name": "test_renamed.mp3", "confidence": 0.5},
                ],
                "confidence": 0.5,
            }

            proposal = await generator.generate_proposal("test.mp3", templates=[template])

            assert proposal.original_filename == "test.mp3"
            assert "template" in proposal.explanation.lower()

    @pytest.mark.asyncio
    async def test_apply_template(self) -> None:
        """Test template application."""
        generator = ProposalGenerator()

        template = NamingTemplate(
            id="t1",
            name="Test",
            pattern="${artist} - ${date} - ${venue}",
            user_id="user1",
            created_at=datetime.now(UTC),
            usage_count=0,
        )

        context = {
            "artist": "Phish",
            "date": "2024-01-29",
            "venue": "Madison Square Garden",
        }

        result = generator.apply_template("test.mp3", template, context)
        # The apply_template function cleans up the formatting
        assert "Phish" in result
        assert "2024" in result
        assert "Madison Square Garden" in result


class TestConfidenceScorer:
    """Test the confidence scorer."""

    def test_calculate_confidence_basic(self) -> None:
        """Test basic confidence calculation."""
        scorer = ConfidenceScorer()

        confidence = scorer.calculate_confidence(
            ml_confidence=0.8,
            pattern_frequency={"artist_title": 100},
            pattern_name="artist_title",
        )

        assert 0 <= confidence <= 1
        assert confidence >= 0.8  # Should be boosted by frequency

    def test_normalize_score(self) -> None:
        """Test score normalization."""
        scorer = ConfidenceScorer()

        # Test clamping
        assert scorer.normalize_score(1.5) == 1.0
        assert scorer.normalize_score(-0.5) == 0.0
        assert scorer.normalize_score(0.5) == 0.5

    def test_factor_user_feedback(self) -> None:
        """Test user feedback factoring."""
        scorer = ConfidenceScorer()

        # High approval rate should boost confidence
        positive_feedback = {
            "approval_rate": 0.8,  # 80% approval
            "total_feedback_count": 100,
        }
        boosted = scorer.factor_user_feedback(0.5, positive_feedback)
        assert boosted > 0.5  # Should be higher due to positive feedback

        # Low approval rate should reduce confidence
        negative_feedback = {
            "approval_rate": 0.2,  # 20% approval
            "total_feedback_count": 100,
        }
        reduced = scorer.factor_user_feedback(0.5, negative_feedback)
        assert reduced < 0.5  # Should be lower due to negative feedback


class TestBatchProcessor:
    """Test batch processing capabilities."""

    @pytest.mark.asyncio
    async def test_process_batch(self) -> None:
        """Test batch processing of multiple files."""
        processor = BatchProcessor()

        # Mock the generator
        with patch.object(processor.generator, "generate_proposal") as mock_gen:
            mock_proposal = RenameProposal(
                original_filename="test.mp3",
                proposed_filename="new.mp3",
                confidence_score=0.8,
                explanation="test",
                patterns_used=[],
                alternatives=[],
                conflict_status="none",
            )
            mock_gen.return_value = mock_proposal

            filenames = [f"file{i}.mp3" for i in range(10)]
            result = await processor.process_batch(filenames)

            assert isinstance(result, BatchProcessingResult)
            assert result.total_files == 10
            assert result.successful_count == 10
            assert len(result.proposals) == 10

    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self) -> None:
        """Test batch processing with partial failures."""
        processor = BatchProcessor()

        # Mock generator to fail on some files
        call_count = 0

        async def mock_generate(filename: str, **kwargs: Any) -> RenameProposal:
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise ValueError("Mock error")
            return RenameProposal(
                original_filename=filename,
                proposed_filename=f"new_{filename}",
                confidence_score=0.8,
                explanation="test",
                patterns_used=[],
                alternatives=[],
                conflict_status="none",
            )

        with patch.object(processor.generator, "generate_proposal", mock_generate):
            filenames = [f"file{i}.mp3" for i in range(10)]
            result = await processor.process_batch(filenames)

            assert result.total_files == 10
            assert result.failed_count > 0
            assert result.successful_count < 10
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_process_batch_performance(self) -> None:
        """Test batch processing meets performance requirements."""
        processor = BatchProcessor()

        # Mock fast proposal generation
        with patch.object(processor.generator, "generate_proposal") as mock_gen:
            mock_proposal = RenameProposal(
                original_filename="test.mp3",
                proposed_filename="new.mp3",
                confidence_score=0.8,
                explanation="test",
                patterns_used=[],
                alternatives=[],
                conflict_status="none",
            )
            mock_gen.return_value = mock_proposal

            # Process 100 files
            filenames = [f"file{i}.mp3" for i in range(100)]

            start = time.time()
            result = await processor.process_batch(filenames)
            elapsed = time.time() - start

            # Should process 100 files in < 5 seconds
            assert elapsed < 5
            assert result.successful_count == 100


class TestConflictResolver:
    """Test conflict detection and resolution."""

    def test_validate_filename(self) -> None:
        """Test filename validation."""
        assert validate_filename("valid_file.mp3") is True
        assert validate_filename("CON") is False  # Windows reserved
        assert validate_filename("file<>name.mp3") is False  # Invalid chars
        assert validate_filename("a" * 256) is False  # Too long

    def test_check_duplicate(self) -> None:
        """Test duplicate detection."""
        assert check_duplicate("File.mp3", "file.mp3") is True
        assert check_duplicate("file.mp3", "different.mp3") is False

    def test_generate_unique_name(self) -> None:
        """Test unique name generation."""
        existing = ["file.mp3", "file (1).mp3"]
        unique = generate_unique_name("file.mp3", existing)
        assert unique == "file (2).mp3"

    def test_detect_conflicts(self) -> None:
        """Test conflict detection."""
        existing = ["existing.mp3"]
        conflict = detect_conflicts("existing.mp3", existing)
        assert conflict is not None
        assert conflict.strategy == "append_number"

        no_conflict = detect_conflicts("new.mp3", existing)
        assert no_conflict is None

    def test_resolve_conflict(self) -> None:
        """Test conflict resolution strategies."""
        existing = ["file.mp3"]

        # Test append_number strategy
        resolved = resolve_conflict("file.mp3", existing, "append_number")
        assert resolved == "file (1).mp3"

        # Test skip strategy
        resolved = resolve_conflict("file.mp3", existing, "skip")
        assert resolved == "file.mp3"

        # Test replace strategy
        resolved = resolve_conflict("file.mp3", existing, "replace")
        assert resolved == "file.mp3"


class TestExplainer:
    """Test the explanation system."""

    def test_generate_explanation(self) -> None:
        """Test explanation generation."""
        explainer = RenameExplainer()

        proposal = RenameProposal(
            original_filename="test.mp3",
            proposed_filename="Artist - Song.mp3",
            confidence_score=0.85,
            explanation="Based on patterns",
            patterns_used=["artist_title"],
            alternatives=["Song - Artist.mp3"],
            conflict_status="none",
        )

        explanation = explainer.generate_explanation(proposal)

        assert "test.mp3" in explanation
        assert "Artist - Song.mp3" in explanation
        assert "85%" in explanation

    def test_explain_confidence_factors(self) -> None:
        """Test confidence factor explanation."""
        explainer = RenameExplainer()

        factors = {
            "ml_confidence": 0.8,
            "pattern_frequency": 0.9,
            "user_feedback": 0.7,
        }

        explanation = explainer.explain_confidence_factors(0.85, factors)

        assert "confidence" in explanation.lower()
        assert any(str(v) in explanation or f"{v * 100:.0f}" in explanation for v in factors.values())


class TestTemplateManager:
    """Test template management."""

    @pytest.mark.asyncio
    async def test_save_and_get_template(self) -> None:
        """Test saving and retrieving templates."""
        manager = TemplateManager()

        template = NamingTemplate(
            id="test1",
            name="Test Template",
            pattern="{artist} - {title}",
            user_id="user1",
            created_at=datetime.now(UTC),
            usage_count=0,
        )

        template_id = await manager.save_template(template)
        assert template_id == "test1"

        retrieved = await manager.get_template("test1")
        assert retrieved is not None
        assert retrieved.name == "Test Template"

    @pytest.mark.asyncio
    async def test_get_user_templates(self) -> None:
        """Test retrieving user templates."""
        manager = TemplateManager()

        # Save multiple templates
        for i in range(3):
            template = NamingTemplate(
                id=f"t{i}",
                name=f"Template {i}",
                pattern="{artist}",
                user_id="user1",
                created_at=datetime.now(UTC),
                usage_count=i,
            )
            await manager.save_template(template)

        templates = await manager.get_user_templates("user1")
        assert len(templates) == 3

    def test_validate_template_pattern(self) -> None:
        """Test template pattern validation."""
        manager = TemplateManager()

        assert manager.validate_template_pattern("{artist} - {title}") is True
        assert manager.validate_template_pattern("{artist") is False  # Unbalanced
        assert manager.validate_template_pattern("{}") is False  # Empty variable
        assert manager.validate_template_pattern("{invalid-var}") is False

    def test_parse_template_variables(self) -> None:
        """Test template variable parsing."""
        manager = TemplateManager()

        variables = manager.parse_template_variables("{artist} - {date} - {venue}")
        assert set(variables) == {"artist", "date", "venue"}

    def test_get_default_templates(self) -> None:
        """Test getting default templates."""
        manager = TemplateManager()

        defaults = manager.get_default_templates()
        assert len(defaults) > 0
        assert all(isinstance(t, NamingTemplate) for t in defaults)


class TestProposalCache:
    """Test the caching layer."""

    @pytest.mark.asyncio
    async def test_cache_and_retrieve(self) -> None:
        """Test caching and retrieving proposals."""
        cache = ProposalCache(use_redis=False)  # Use mock implementation

        proposal = RenameProposal(
            original_filename="test.mp3",
            proposed_filename="new.mp3",
            confidence_score=0.8,
            explanation="test",
            patterns_used=[],
            alternatives=[],
            conflict_status="none",
        )

        success = await cache.cache_proposal("test_key", proposal)
        assert success is True

        retrieved = await cache.get_cached_proposal("test_key")
        assert retrieved is not None
        assert retrieved.original_filename == "test.mp3"

    @pytest.mark.asyncio
    async def test_cache_invalidation(self) -> None:
        """Test cache invalidation."""
        cache = ProposalCache(use_redis=False)

        proposal = RenameProposal(
            original_filename="test.mp3",
            proposed_filename="new.mp3",
            confidence_score=0.8,
            explanation="test",
            patterns_used=[],
            alternatives=[],
            conflict_status="none",
        )

        # Cache multiple entries
        await cache.cache_proposal("key1", proposal)
        await cache.cache_proposal("key2", proposal)

        # Invalidate all
        count = await cache.invalidate_cache()
        assert count == 2

        # Verify cache is empty
        assert await cache.get_cached_proposal("key1") is None
        assert await cache.get_cached_proposal("key2") is None

    @pytest.mark.asyncio
    async def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = ProposalCache(use_redis=False)

        proposal = RenameProposal(
            original_filename="test.mp3",
            proposed_filename="new.mp3",
            confidence_score=0.8,
            explanation="test",
            patterns_used=[],
            alternatives=[],
            conflict_status="none",
        )

        # Generate some cache activity
        await cache.cache_proposal("key1", proposal)
        await cache.get_cached_proposal("key1")  # Hit
        await cache.get_cached_proposal("key2")  # Miss

        stats = await cache.get_cache_stats()

        assert stats["total_requests"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["hit_rate"] == 50.0
