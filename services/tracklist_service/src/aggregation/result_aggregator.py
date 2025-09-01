"""Result aggregation for batch processing operations."""

import json
import logging
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from redis import Redis

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Types of aggregation operations."""

    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    LIST = "list"
    SET = "set"
    HISTOGRAM = "histogram"
    PERCENTILE = "percentile"
    CUSTOM = "custom"


class DataFormat(Enum):
    """Output data format types."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class AggregationRule:
    """Rule for aggregating specific fields."""

    field_path: str
    aggregation_type: AggregationType
    output_name: str
    filter_condition: Callable[[Any], bool] | None = None
    transform: Callable[[Any], Any] | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of aggregation operation."""

    batch_id: str
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    aggregated_data: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    processing_time: float
    export_formats: list[str] = field(default_factory=list)


class ResultAggregator:
    """Aggregates and analyzes batch processing results."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        storage_path: str = "/tmp/aggregated_results",
    ):
        """Initialize result aggregator.

        Args:
            redis_host: Redis host for result storage
            redis_port: Redis port
            storage_path: Path for file exports
        """
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.storage_path = storage_path

        # Result storage
        self.batch_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.aggregation_rules: dict[str, list[AggregationRule]] = defaultdict(list)
        self.aggregated_results: dict[str, AggregationResult] = {}

        # Custom aggregation functions
        self.custom_aggregators: dict[str, Callable[..., Any]] = {}

    def add_result(self, batch_id: str, job_id: str, result: dict[str, Any]) -> None:
        """Add job result to batch.

        Args:
            batch_id: Batch identifier
            job_id: Job identifier
            result: Job result data
        """
        # Store in memory
        self.batch_results[batch_id].append({"job_id": job_id, "timestamp": datetime.now(UTC).isoformat(), **result})

        # Persist to Redis
        key = f"batch_result:{batch_id}:{job_id}"
        self.redis.hset(
            key,
            mapping={
                "result": json.dumps(result),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        self.redis.expire(key, 86400)  # 24 hour TTL

    def add_aggregation_rule(
        self,
        batch_id: str,
        field_path: str,
        aggregation_type: AggregationType,
        output_name: str | None = None,
        **options: Any,
    ) -> None:
        """Add aggregation rule for batch.

        Args:
            batch_id: Batch identifier
            field_path: Path to field in result (e.g., "data.price")
            aggregation_type: Type of aggregation
            output_name: Name for aggregated value
            **options: Additional options for aggregation
        """
        rule = AggregationRule(
            field_path=field_path,
            aggregation_type=aggregation_type,
            output_name=output_name or f"{field_path}_{aggregation_type.value}",
            options=options,
        )
        self.aggregation_rules[batch_id].append(rule)

    def register_custom_aggregator(self, name: str, func: Callable[[list[Any]], Any]) -> None:
        """Register custom aggregation function.

        Args:
            name: Aggregator name
            func: Aggregation function
        """
        self.custom_aggregators[name] = func

    async def aggregate_batch(self, batch_id: str, include_failed: bool = False) -> AggregationResult:
        """Aggregate results for entire batch.

        Args:
            batch_id: Batch identifier
            include_failed: Include failed jobs in aggregation

        Returns:
            Aggregation result
        """
        start_time = datetime.now(UTC)

        # Load results from Redis if not in memory
        if batch_id not in self.batch_results:
            await self._load_batch_results(batch_id)

        results = self.batch_results.get(batch_id, [])
        if not results:
            logger.warning(f"No results found for batch {batch_id}")
            return self._empty_result(batch_id)

        # Filter failed jobs if requested
        if not include_failed:
            results = [r for r in results if not r.get("error")]

        # Apply aggregation rules
        aggregated = {}
        rules = self.aggregation_rules.get(batch_id, [])

        if not rules:
            # Default aggregations if no rules specified
            rules = self._get_default_rules(results)

        for rule in rules:
            value = await self._apply_aggregation_rule(results, rule)
            aggregated[rule.output_name] = value

        # Calculate metadata
        processing_time = (datetime.now(UTC) - start_time).total_seconds()

        result = AggregationResult(
            batch_id=batch_id,
            total_jobs=len(self.batch_results.get(batch_id, [])),
            successful_jobs=len(results),
            failed_jobs=len(self.batch_results.get(batch_id, [])) - len(results),
            aggregated_data=aggregated,
            metadata={
                "aggregation_rules": len(rules),
                "processing_time": processing_time,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            created_at=datetime.now(UTC),
            processing_time=processing_time,
        )

        # Store result
        self.aggregated_results[batch_id] = result
        await self._persist_aggregation(result)

        return result

    async def _apply_aggregation_rule(self, results: list[dict[str, Any]], rule: AggregationRule) -> Any:
        """Apply single aggregation rule.

        Args:
            results: List of results
            rule: Aggregation rule

        Returns:
            Aggregated value
        """
        # Extract values from field path
        values = []
        for result in results:
            value = self._extract_field_value(result, rule.field_path)

            # Apply filter if specified
            if rule.filter_condition and not rule.filter_condition(value):
                continue

            # Apply transform if specified
            if rule.transform:
                value = rule.transform(value)

            if value is not None:
                values.append(value)

        if not values:
            return None

        # Apply aggregation
        if rule.aggregation_type == AggregationType.COUNT:
            return len(values)
        if rule.aggregation_type == AggregationType.SUM:
            return sum(values)
        if rule.aggregation_type == AggregationType.AVERAGE:
            return statistics.mean(values)
        if rule.aggregation_type == AggregationType.MEDIAN:
            return statistics.median(values)
        if rule.aggregation_type == AggregationType.MIN:
            return min(values)
        if rule.aggregation_type == AggregationType.MAX:
            return max(values)
        if rule.aggregation_type == AggregationType.LIST:
            return values
        if rule.aggregation_type == AggregationType.SET:
            return list(set(values))
        if rule.aggregation_type == AggregationType.HISTOGRAM:
            return self._create_histogram(values, rule.options)
        if rule.aggregation_type == AggregationType.PERCENTILE:
            percentile = rule.options.get("percentile", 50)
            return self._calculate_percentile(values, percentile)
        if rule.aggregation_type == AggregationType.CUSTOM:
            func_name = rule.options.get("function")
            if func_name and isinstance(func_name, str) and func_name in self.custom_aggregators:
                return self.custom_aggregators[func_name](values)

        return None

    def _extract_field_value(self, data: dict[str, Any], field_path: str) -> Any | None:
        """Extract value from nested dictionary using dot notation.

        Args:
            data: Source dictionary
            field_path: Dot-separated path (e.g., "data.items.price")

        Returns:
            Extracted value or None
        """
        parts = field_path.split(".")
        value: Any = data

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                if idx < len(value):
                    value = value[idx]
                else:
                    return None
            else:
                return None

        return value

    def _create_histogram(self, values: list[float], options: dict[str, Any]) -> dict[str, Any]:
        """Create histogram from values.

        Args:
            values: List of numeric values
            options: Histogram options

        Returns:
            Histogram data
        """
        bins = options.get("bins", 10)

        # Calculate histogram
        hist, edges = pd.cut(values, bins=bins, retbins=True)
        counts = hist.value_counts().sort_index()

        return {
            "bins": [str(interval) for interval in counts.index],
            "counts": counts.tolist(),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
        }

    def _calculate_percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        sorted_values = sorted(values)
        index = (len(sorted_values) - 1) * (percentile / 100)

        if index.is_integer():
            return sorted_values[int(index)]
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        weight = index - int(index)
        return lower + (upper - lower) * weight

    def _get_default_rules(self, results: list[dict[str, Any]]) -> list[AggregationRule]:
        """Generate default aggregation rules based on data structure.

        Args:
            results: Sample results

        Returns:
            List of default rules
        """
        rules = []

        # Add count rule
        rules.append(
            AggregationRule(
                field_path="job_id",
                aggregation_type=AggregationType.COUNT,
                output_name="total_count",
            )
        )

        # Analyze first result for numeric fields
        if results:
            sample = results[0]
            numeric_fields = self._find_numeric_fields(sample)

            for field in numeric_fields:
                # Add average and sum for numeric fields
                rules.append(
                    AggregationRule(
                        field_path=field,
                        aggregation_type=AggregationType.AVERAGE,
                        output_name=f"{field}_avg",
                    )
                )
                rules.append(
                    AggregationRule(
                        field_path=field,
                        aggregation_type=AggregationType.SUM,
                        output_name=f"{field}_sum",
                    )
                )

        return rules

    def _find_numeric_fields(self, data: dict[str, Any], prefix: str = "") -> list[str]:
        """Find all numeric fields in nested dictionary.

        Args:
            data: Dictionary to analyze
            prefix: Field path prefix

        Returns:
            List of numeric field paths
        """
        numeric_fields = []

        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, int | float):
                numeric_fields.append(field_path)
            elif isinstance(value, dict):
                # Recursively search nested dictionaries
                numeric_fields.extend(self._find_numeric_fields(value, field_path))

        return numeric_fields

    async def export_results(self, batch_id: str, format: DataFormat, file_path: str | None = None) -> str:
        """Export aggregated results to file.

        Args:
            batch_id: Batch identifier
            format: Export format
            file_path: Output file path

        Returns:
            Path to exported file
        """
        if batch_id not in self.aggregated_results:
            await self.aggregate_batch(batch_id)

        result = self.aggregated_results[batch_id]

        # Generate file path if not provided
        if not file_path:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            file_path = f"{self.storage_path}/{batch_id}_{timestamp}.{format.value}"

        # Export based on format
        if format == DataFormat.JSON:
            await self._export_json(result, file_path)
        elif format == DataFormat.CSV:
            await self._export_csv(result, file_path)
        elif format == DataFormat.PARQUET:
            await self._export_parquet(result, file_path)
        elif format == DataFormat.EXCEL:
            await self._export_excel(result, file_path)
        elif format == DataFormat.HTML:
            await self._export_html(result, file_path)
        elif format == DataFormat.MARKDOWN:
            await self._export_markdown(result, file_path)

        # Track export
        result.export_formats.append(format.value)

        return file_path

    async def _export_json(self, result: AggregationResult, file_path: str) -> None:
        """Export to JSON format."""
        with Path(file_path).open("w") as f:
            json.dump(
                {
                    "batch_id": result.batch_id,
                    "summary": {
                        "total_jobs": result.total_jobs,
                        "successful_jobs": result.successful_jobs,
                        "failed_jobs": result.failed_jobs,
                    },
                    "aggregated_data": result.aggregated_data,
                    "metadata": result.metadata,
                },
                f,
                indent=2,
                default=str,
            )

    async def _export_csv(self, result: AggregationResult, file_path: str) -> None:
        """Export to CSV format."""
        # Flatten aggregated data for CSV
        rows = []
        for key, value in result.aggregated_data.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    rows.append({"metric": f"{key}_{i}", "value": item})
            else:
                rows.append({"metric": key, "value": value})

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(file_path, index=False)

    async def _export_parquet(self, result: AggregationResult, file_path: str) -> None:
        """Export to Parquet format."""
        result_df = pd.DataFrame([result.aggregated_data])
        result_df.to_parquet(file_path, index=False)

    async def _export_excel(self, result: AggregationResult, file_path: str) -> None:
        """Export to Excel format."""
        with pd.ExcelWriter(file_path) as writer:
            # Summary sheet
            summary_df = pd.DataFrame(
                [
                    {
                        "Batch ID": result.batch_id,
                        "Total Jobs": result.total_jobs,
                        "Successful": result.successful_jobs,
                        "Failed": result.failed_jobs,
                        "Processing Time": result.processing_time,
                    }
                ]
            )
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Aggregated data sheet
            data_df = pd.DataFrame([result.aggregated_data])
            data_df.to_excel(writer, sheet_name="Aggregated Data", index=False)

    async def _export_html(self, result: AggregationResult, file_path: str) -> None:
        """Export to HTML format."""
        html = f"""
        <html>
        <head>
            <title>Batch {result.batch_id} Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Batch Processing Results</h1>
            <h2>Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Batch ID</td><td>{result.batch_id}</td></tr>
                <tr><td>Total Jobs</td><td>{result.total_jobs}</td></tr>
                <tr><td>Successful</td><td>{result.successful_jobs}</td></tr>
                <tr><td>Failed</td><td>{result.failed_jobs}</td></tr>
                <tr><td>Processing Time</td><td>{result.processing_time:.2f}s</td></tr>
            </table>
            <h2>Aggregated Data</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """

        for key, value in result.aggregated_data.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html += """
            </table>
        </body>
        </html>
        """

        with Path(file_path).open("w") as f:
            f.write(html)

    async def _export_markdown(self, result: AggregationResult, file_path: str) -> None:
        """Export to Markdown format."""
        md = f"""# Batch Processing Results

## Batch: {result.batch_id}

### Summary
| Metric | Value |
|--------|-------|
| Total Jobs | {result.total_jobs} |
| Successful | {result.successful_jobs} |
| Failed | {result.failed_jobs} |
| Processing Time | {result.processing_time:.2f}s |

### Aggregated Data
| Metric | Value |
|--------|-------|
"""

        for key, value in result.aggregated_data.items():
            md += f"| {key} | {value} |\n"

        with Path(file_path).open("w") as f:
            f.write(md)

    async def _load_batch_results(self, batch_id: str) -> None:
        """Load batch results from Redis.

        Args:
            batch_id: Batch identifier
        """
        pattern = f"batch_result:{batch_id}:*"
        results = []

        for key in self.redis.scan_iter(match=pattern):
            data = await self.redis.hgetall(key)
            if data and "result" in data:
                result = json.loads(data["result"])
                job_id = key.split(":")[-1]
                results.append({"job_id": job_id, "timestamp": data.get("timestamp"), **result})

        self.batch_results[batch_id] = results

    async def _persist_aggregation(self, result: AggregationResult) -> None:
        """Persist aggregation result to Redis.

        Args:
            result: Aggregation result
        """
        key = f"aggregation:{result.batch_id}"
        self.redis.hset(
            key,
            mapping={
                "total_jobs": result.total_jobs,
                "successful_jobs": result.successful_jobs,
                "failed_jobs": result.failed_jobs,
                "data": json.dumps(result.aggregated_data, default=str),
                "metadata": json.dumps(result.metadata, default=str),
                "created_at": result.created_at.isoformat(),
            },
        )
        self.redis.expire(key, 86400)  # 24 hour TTL

    def _empty_result(self, batch_id: str) -> AggregationResult:
        """Create empty aggregation result.

        Args:
            batch_id: Batch identifier

        Returns:
            Empty result
        """
        return AggregationResult(
            batch_id=batch_id,
            total_jobs=0,
            successful_jobs=0,
            failed_jobs=0,
            aggregated_data={},
            metadata={"error": "No results found"},
            created_at=datetime.now(UTC),
            processing_time=0.0,
        )

    def get_aggregation_status(self, batch_id: str) -> dict[str, Any]:
        """Get aggregation status for batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Status information
        """
        if batch_id in self.aggregated_results:
            result = self.aggregated_results[batch_id]
            return {
                "status": "completed",
                "total_jobs": result.total_jobs,
                "successful_jobs": result.successful_jobs,
                "failed_jobs": result.failed_jobs,
                "created_at": result.created_at.isoformat(),
                "export_formats": result.export_formats,
            }
        if batch_id in self.batch_results:
            return {
                "status": "pending",
                "collected_results": len(self.batch_results[batch_id]),
                "rules_defined": len(self.aggregation_rules.get(batch_id, [])),
            }
        return {"status": "not_found"}
