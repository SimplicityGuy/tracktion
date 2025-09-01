"""Unit tests for result aggregation system."""

from datetime import UTC, datetime
from unittest.mock import Mock, mock_open, patch

import pytest

from services.tracklist_service.src.aggregation.result_aggregator import (
    AggregationResult,
    AggregationRule,
    AggregationType,
    DataFormat,
    ResultAggregator,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = Mock()
    redis_mock.hset = Mock()
    redis_mock.hgetall = Mock(return_value={})
    redis_mock.expire = Mock()
    redis_mock.scan_iter = Mock(return_value=[])
    return redis_mock


@pytest.fixture
def aggregator(mock_redis):
    """Create ResultAggregator instance with mock Redis."""
    with patch(
        "services.tracklist_service.src.aggregation.result_aggregator.Redis",
        return_value=mock_redis,
    ):
        agg = ResultAggregator()
        yield agg


class TestAggregationType:
    """Test AggregationType enum."""

    def test_aggregation_types(self):
        """Test aggregation type values."""
        assert AggregationType.COUNT.value == "count"
        assert AggregationType.SUM.value == "sum"
        assert AggregationType.AVERAGE.value == "average"
        assert AggregationType.MEDIAN.value == "median"
        assert AggregationType.HISTOGRAM.value == "histogram"
        assert AggregationType.CUSTOM.value == "custom"


class TestDataFormat:
    """Test DataFormat enum."""

    def test_data_formats(self):
        """Test data format values."""
        assert DataFormat.JSON.value == "json"
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.PARQUET.value == "parquet"
        assert DataFormat.HTML.value == "html"
        assert DataFormat.MARKDOWN.value == "markdown"


class TestAggregationRule:
    """Test AggregationRule dataclass."""

    def test_initialization(self):
        """Test rule initialization."""
        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.AVERAGE,
            output_name="avg_price",
        )

        assert rule.field_path == "data.price"
        assert rule.aggregation_type == AggregationType.AVERAGE
        assert rule.output_name == "avg_price"
        assert rule.filter_condition is None
        assert rule.transform is None

    def test_with_filter_and_transform(self):
        """Test rule with filter and transform."""
        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.SUM,
            output_name="total_price",
            filter_condition=lambda x: x > 0,
            transform=lambda x: x * 1.1,
        )

        assert rule.filter_condition(10) is True
        assert rule.filter_condition(-5) is False
        assert abs(rule.transform(100) - 110) < 0.01  # Float comparison with tolerance


class TestResultAggregator:
    """Test ResultAggregator class."""

    def test_initialization(self, mock_redis):
        """Test aggregator initialization."""
        with patch(
            "services.tracklist_service.src.aggregation.result_aggregator.Redis",
            return_value=mock_redis,
        ):
            agg = ResultAggregator()

        assert agg.redis == mock_redis
        assert agg.batch_results == {}
        assert agg.aggregation_rules == {}
        assert agg.custom_aggregators == {}

    def test_add_result(self, aggregator, mock_redis):
        """Test adding result to batch."""
        aggregator.add_result(batch_id="batch-123", job_id="job-1", result={"data": {"price": 100}})

        # Check result stored in memory
        assert "batch-123" in aggregator.batch_results
        assert len(aggregator.batch_results["batch-123"]) == 1
        assert aggregator.batch_results["batch-123"][0]["job_id"] == "job-1"

        # Check Redis persistence
        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()

    def test_add_aggregation_rule(self, aggregator):
        """Test adding aggregation rule."""
        aggregator.add_aggregation_rule(
            batch_id="batch-123",
            field_path="data.price",
            aggregation_type=AggregationType.AVERAGE,
            output_name="avg_price",
        )

        assert "batch-123" in aggregator.aggregation_rules
        assert len(aggregator.aggregation_rules["batch-123"]) == 1

        rule = aggregator.aggregation_rules["batch-123"][0]
        assert rule.field_path == "data.price"
        assert rule.aggregation_type == AggregationType.AVERAGE

    def test_register_custom_aggregator(self, aggregator):
        """Test registering custom aggregation function."""

        def custom_func(values):
            return sum(values) * 2

        aggregator.register_custom_aggregator("double_sum", custom_func)

        assert "double_sum" in aggregator.custom_aggregators
        assert aggregator.custom_aggregators["double_sum"]([1, 2, 3]) == 12

    def test_extract_field_value(self, aggregator):
        """Test extracting field value from nested dict."""
        data = {"data": {"items": [{"price": 100}, {"price": 200}], "total": 300}}

        # Test nested field
        assert aggregator._extract_field_value(data, "data.total") == 300

        # Test array access
        assert aggregator._extract_field_value(data, "data.items.0.price") == 100

        # Test non-existent field
        assert aggregator._extract_field_value(data, "data.missing") is None

    @pytest.mark.asyncio
    async def test_apply_aggregation_rule_count(self, aggregator):
        """Test COUNT aggregation."""
        results = [
            {"data": {"price": 100}},
            {"data": {"price": 200}},
            {"data": {"price": 300}},
        ]

        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.COUNT,
            output_name="count",
        )

        value = await aggregator._apply_aggregation_rule(results, rule)
        assert value == 3

    @pytest.mark.asyncio
    async def test_apply_aggregation_rule_sum(self, aggregator):
        """Test SUM aggregation."""
        results = [
            {"data": {"price": 100}},
            {"data": {"price": 200}},
            {"data": {"price": 300}},
        ]

        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.SUM,
            output_name="total",
        )

        value = await aggregator._apply_aggregation_rule(results, rule)
        assert value == 600

    @pytest.mark.asyncio
    async def test_apply_aggregation_rule_average(self, aggregator):
        """Test AVERAGE aggregation."""
        results = [
            {"data": {"price": 100}},
            {"data": {"price": 200}},
            {"data": {"price": 300}},
        ]

        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.AVERAGE,
            output_name="avg",
        )

        value = await aggregator._apply_aggregation_rule(results, rule)
        assert value == 200

    @pytest.mark.asyncio
    async def test_apply_aggregation_rule_with_filter(self, aggregator):
        """Test aggregation with filter."""
        results = [
            {"data": {"price": 100}},
            {"data": {"price": -50}},
            {"data": {"price": 200}},
        ]

        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.SUM,
            output_name="positive_sum",
            filter_condition=lambda x: x > 0,
        )

        value = await aggregator._apply_aggregation_rule(results, rule)
        assert value == 300  # Only positive values

    @pytest.mark.asyncio
    async def test_apply_aggregation_rule_with_transform(self, aggregator):
        """Test aggregation with transform."""
        results = [{"data": {"price": 100}}, {"data": {"price": 200}}]

        rule = AggregationRule(
            field_path="data.price",
            aggregation_type=AggregationType.SUM,
            output_name="total_with_tax",
            transform=lambda x: x * 1.1,  # Add 10% tax
        )

        value = await aggregator._apply_aggregation_rule(results, rule)
        assert abs(value - 330) < 0.01  # (100 + 200) * 1.1 with tolerance

    @pytest.mark.asyncio
    async def test_aggregate_batch(self, aggregator):
        """Test batch aggregation."""
        # Add results
        aggregator.batch_results["batch-123"] = [
            {"job_id": "job-1", "data": {"price": 100}},
            {"job_id": "job-2", "data": {"price": 200}},
            {"job_id": "job-3", "data": {"price": 300}},
        ]

        # Add rules
        aggregator.add_aggregation_rule("batch-123", "data.price", AggregationType.AVERAGE, "avg_price")
        aggregator.add_aggregation_rule("batch-123", "data.price", AggregationType.SUM, "total_price")

        result = await aggregator.aggregate_batch("batch-123")

        assert result.batch_id == "batch-123"
        assert result.total_jobs == 3
        assert result.successful_jobs == 3
        assert result.aggregated_data["avg_price"] == 200
        assert result.aggregated_data["total_price"] == 600

    @pytest.mark.asyncio
    async def test_aggregate_batch_with_failed_jobs(self, aggregator):
        """Test aggregation excluding failed jobs."""
        aggregator.batch_results["batch-123"] = [
            {"job_id": "job-1", "data": {"price": 100}},
            {"job_id": "job-2", "error": "Failed", "data": {}},
            {"job_id": "job-3", "data": {"price": 300}},
        ]

        aggregator.add_aggregation_rule("batch-123", "data.price", AggregationType.AVERAGE)

        result = await aggregator.aggregate_batch("batch-123", include_failed=False)

        assert result.total_jobs == 3
        assert result.successful_jobs == 2
        assert result.failed_jobs == 1

    def test_find_numeric_fields(self, aggregator):
        """Test finding numeric fields in nested dict."""
        data = {
            "id": "123",
            "data": {
                "price": 100.5,
                "quantity": 5,
                "name": "Product",
                "metrics": {"views": 1000, "clicks": 50},
            },
        }

        fields = aggregator._find_numeric_fields(data)

        assert "data.price" in fields
        assert "data.quantity" in fields
        assert "data.metrics.views" in fields
        assert "data.metrics.clicks" in fields
        assert "id" not in fields  # String field

    def test_calculate_percentile(self, aggregator):
        """Test percentile calculation."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # 50th percentile (median)
        assert aggregator._calculate_percentile(values, 50) == 5.5

        # 25th percentile
        assert aggregator._calculate_percentile(values, 25) == 3.25

        # 75th percentile
        assert aggregator._calculate_percentile(values, 75) == 7.75

    @pytest.mark.asyncio
    async def test_export_json(self, aggregator):
        """Test JSON export."""
        result = AggregationResult(
            batch_id="batch-123",
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            aggregated_data={"avg_price": 200, "total": 1600},
            metadata={"timestamp": "2024-01-01"},
            created_at=datetime.now(UTC),
            processing_time=1.5,
        )

        with patch("builtins.open", mock_open()) as mock_file:
            await aggregator._export_json(result, "/tmp/test.json")

            # Check file was opened for writing
            mock_file.assert_called_once_with("/tmp/test.json", "w")

            # Check write was called
            handle = mock_file()
            assert handle.write.called or handle.__enter__().write.called

    @pytest.mark.asyncio
    async def test_export_csv(self, aggregator):
        """Test CSV export."""
        result = AggregationResult(
            batch_id="batch-123",
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            aggregated_data={"avg_price": 200, "total": 1600},
            metadata={},
            created_at=datetime.now(UTC),
            processing_time=1.5,
        )

        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            await aggregator._export_csv(result, "/tmp/test.csv")
            mock_to_csv.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_html(self, aggregator):
        """Test HTML export."""
        result = AggregationResult(
            batch_id="batch-123",
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            aggregated_data={"avg_price": 200},
            metadata={},
            created_at=datetime.now(UTC),
            processing_time=1.5,
        )

        with patch("builtins.open", mock_open()) as mock_file:
            await aggregator._export_html(result, "/tmp/test.html")

            mock_file.assert_called_once_with("/tmp/test.html", "w")
            handle = mock_file()

            # Check HTML content was written
            write_calls = handle.write.call_args_list
            if write_calls:
                html_content = write_calls[0][0][0]
                assert "<html>" in html_content
                assert "batch-123" in html_content

    @pytest.mark.asyncio
    async def test_export_markdown(self, aggregator):
        """Test Markdown export."""
        result = AggregationResult(
            batch_id="batch-123",
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            aggregated_data={"avg_price": 200},
            metadata={},
            created_at=datetime.now(UTC),
            processing_time=1.5,
        )

        with patch("builtins.open", mock_open()) as mock_file:
            await aggregator._export_markdown(result, "/tmp/test.md")

            mock_file.assert_called_once_with("/tmp/test.md", "w")
            handle = mock_file()

            # Check Markdown content
            write_calls = handle.write.call_args_list
            if write_calls:
                md_content = write_calls[0][0][0]
                assert "# Batch Processing Results" in md_content
                assert "batch-123" in md_content

    def test_get_aggregation_status(self, aggregator):
        """Test getting aggregation status."""
        # No data
        status = aggregator.get_aggregation_status("batch-999")
        assert status["status"] == "not_found"

        # With results but not aggregated
        aggregator.batch_results["batch-123"] = [{"job_id": "job-1"}]
        status = aggregator.get_aggregation_status("batch-123")
        assert status["status"] == "pending"
        assert status["collected_results"] == 1

        # With aggregated results
        aggregator.aggregated_results["batch-456"] = AggregationResult(
            batch_id="batch-456",
            total_jobs=10,
            successful_jobs=8,
            failed_jobs=2,
            aggregated_data={},
            metadata={},
            created_at=datetime.now(UTC),
            processing_time=1.0,
        )
        status = aggregator.get_aggregation_status("batch-456")
        assert status["status"] == "completed"
        assert status["total_jobs"] == 10
