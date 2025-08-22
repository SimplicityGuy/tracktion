"""Monitoring dashboard for resilient scraping system."""

import asyncio
import json
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from services.tracklist_service.src.monitoring.alert_manager import AlertManager, AlertSeverity
from services.tracklist_service.src.monitoring.structure_monitor import StructureMonitor
from services.tracklist_service.src.cache.fallback_cache import FallbackCache


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: datetime
    parser_health: Dict[str, Any]
    cache_stats: Dict[str, Any]
    alert_summary: Dict[str, int]
    recent_alerts: List[Dict[str, Any]]
    structure_changes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class MonitoringDashboard:
    """Real-time monitoring dashboard for resilient scraping system."""

    def __init__(
        self,
        alert_manager: AlertManager,
        structure_monitor: StructureMonitor,
        fallback_cache: FallbackCache,
    ):
        """Initialize dashboard.

        Args:
            alert_manager: Alert management system
            structure_monitor: Structure change monitor
            fallback_cache: Fallback cache system
        """
        self.alert_manager = alert_manager
        self.structure_monitor = structure_monitor
        self.fallback_cache = fallback_cache
        self._metrics_history: List[SystemMetrics] = []
        self._max_history = 1000  # Keep last 1000 metric snapshots

    async def get_system_metrics(self, page_types: Optional[List[str]] = None) -> SystemMetrics:
        """Get comprehensive system metrics.

        Args:
            page_types: Page types to monitor (if None, uses common types)

        Returns:
            SystemMetrics object with current system state
        """
        if page_types is None:
            page_types = ["tracklist", "artist", "label", "mix"]

        # Collect parser health for each page type
        parser_health = {}
        for page_type in page_types:
            try:
                health = await self.alert_manager.check_parser_health(page_type)
                parser_health[page_type] = {
                    "healthy": health.healthy,
                    "success_rate": health.success_rate,
                    "last_check": health.last_check.isoformat(),
                    "anomaly_count": len(health.anomalies),
                    "failed_extractions": len(health.failed_extractions),
                    "metrics": health.metrics,
                }
            except Exception as e:
                parser_health[page_type] = {
                    "healthy": False,
                    "success_rate": 0.0,
                    "error": str(e),
                }

        # Get cache statistics
        cache_stats = self.fallback_cache.get_cache_stats()

        # Calculate cache performance metrics
        total_requests = cache_stats["hits"] + cache_stats["misses"]
        cache_stats["hit_rate"] = cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        cache_stats["fallback_rate"] = cache_stats["fallback_hits"] / total_requests if total_requests > 0 else 0.0

        # Get alert summary
        alert_summary = self._get_alert_summary()

        # Get recent alerts
        recent_alerts = [alert.to_dict() for alert in self.alert_manager.get_recent_alerts(limit=20)]

        # Get recent structure changes (mock data for now)
        structure_changes = await self._get_recent_structure_changes()

        metrics = SystemMetrics(
            timestamp=datetime.now(UTC),
            parser_health=parser_health,
            cache_stats=cache_stats,
            alert_summary=alert_summary,
            recent_alerts=recent_alerts,
            structure_changes=structure_changes,
        )

        # Store in history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)

        return metrics

    def _get_alert_summary(self) -> Dict[str, int]:
        """Get alert summary by severity."""
        summary = {severity.value: 0 for severity in AlertSeverity}

        # Count alerts from last 24 hours
        recent_alerts = self.alert_manager.get_recent_alerts(limit=1000)
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        for alert in recent_alerts:
            if alert.timestamp > cutoff_time:
                summary[alert.severity.value] += 1

        return summary

    async def _get_recent_structure_changes(self) -> List[Dict[str, Any]]:
        """Get recent structure changes."""
        # This would integrate with structure monitor's change history
        # For now, return mock data
        return [
            {
                "page_type": "tracklist",
                "url": "https://example.com/tracklist/123",
                "detected_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
                "severity": "medium",
                "changes": ["Selector .track-title changed to .title"],
                "requires_manual_review": False,
            },
        ]

    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status.

        Returns:
            Dictionary with overall health assessment
        """
        metrics = await self.get_system_metrics()

        # Calculate overall health score
        health_score = 1.0
        issues = []

        # Check parser health
        unhealthy_parsers = [
            page_type for page_type, health in metrics.parser_health.items() if not health.get("healthy", False)
        ]

        if unhealthy_parsers:
            health_score -= 0.3
            issues.append(f"Unhealthy parsers: {', '.join(unhealthy_parsers)}")

        # Check cache performance
        hit_rate = metrics.cache_stats.get("hit_rate", 0.0)
        if hit_rate < 0.7:
            health_score -= 0.2
            issues.append(f"Low cache hit rate: {hit_rate:.1%}")

        # Check recent alerts
        critical_alerts = metrics.alert_summary.get("critical", 0)
        error_alerts = metrics.alert_summary.get("error", 0)

        if critical_alerts > 0:
            health_score -= 0.4
            issues.append(f"{critical_alerts} critical alerts in last 24h")
        elif error_alerts > 5:
            health_score -= 0.2
            issues.append(f"{error_alerts} error alerts in last 24h")

        # Determine status
        if health_score >= 0.9:
            status = "healthy"
        elif health_score >= 0.7:
            status = "warning"
        elif health_score >= 0.5:
            status = "degraded"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "timestamp": datetime.now(UTC).isoformat(),
            "issues": issues,
            "uptime_metrics": {
                "parsers_healthy": len([h for h in metrics.parser_health.values() if h.get("healthy", False)]),
                "total_parsers": len(metrics.parser_health),
                "cache_hit_rate": hit_rate,
                "alerts_24h": sum(metrics.alert_summary.values()),
            },
        }

    async def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time.

        Args:
            hours: Hours of history to analyze

        Returns:
            Performance trend data
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)

        # Filter metrics history
        recent_metrics = [m for m in self._metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"error": "No metrics data available"}

        # Calculate trends
        trends: Dict[str, Any] = {
            "timespan_hours": hours,
            "data_points": len(recent_metrics),
            "parser_success_rates": {},
            "cache_hit_rates": [],
            "alert_counts": [],
            "timestamps": [],
        }

        for metrics in recent_metrics:
            trends["timestamps"].append(metrics.timestamp.isoformat())
            trends["cache_hit_rates"].append(metrics.cache_stats.get("hit_rate", 0.0))
            trends["alert_counts"].append(sum(metrics.alert_summary.values()))

            # Track parser success rates
            for page_type, health in metrics.parser_health.items():
                if page_type not in trends["parser_success_rates"]:
                    trends["parser_success_rates"][page_type] = []
                trends["parser_success_rates"][page_type].append(health.get("success_rate", 0.0))

        return trends

    async def get_active_issues(self) -> List[Dict[str, Any]]:
        """Get list of active issues requiring attention.

        Returns:
            List of active issues with priority and recommendations
        """
        issues = []
        metrics = await self.get_system_metrics()

        # Check unhealthy parsers
        for page_type, health in metrics.parser_health.items():
            if not health.get("healthy", False):
                issues.append(
                    {
                        "type": "parser_health",
                        "priority": "high" if health.get("success_rate", 0) < 0.5 else "medium",
                        "title": f"Parser health degraded: {page_type}",
                        "description": f"Success rate: {health.get('success_rate', 0):.1%}",
                        "recommendations": [
                            "Check for recent site changes",
                            "Review extraction selectors",
                            "Enable pattern learning",
                            "Consider parser rollback",
                        ],
                        "page_type": page_type,
                    }
                )

        # Check cache performance
        hit_rate = metrics.cache_stats.get("hit_rate", 0.0)
        if hit_rate < 0.7:
            issues.append(
                {
                    "type": "cache_performance",
                    "priority": "medium" if hit_rate > 0.5 else "high",
                    "title": "Low cache hit rate",
                    "description": f"Current hit rate: {hit_rate:.1%}",
                    "recommendations": [
                        "Review TTL settings",
                        "Check data quality scoring",
                        "Monitor Redis performance",
                        "Consider cache warming",
                    ],
                }
            )

        # Check high alert volume
        total_alerts = sum(metrics.alert_summary.values())
        if total_alerts > 20:
            issues.append(
                {
                    "type": "high_alert_volume",
                    "priority": "medium",
                    "title": "High alert volume",
                    "description": f"{total_alerts} alerts in last 24h",
                    "recommendations": [
                        "Review alert thresholds",
                        "Check for recurring issues",
                        "Consider alert suppression",
                        "Investigate root causes",
                    ],
                }
            )

        # Check recent critical alerts
        active_alerts = await self.alert_manager.get_active_alerts()
        critical_active = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]

        for alert in critical_active:
            issues.append(
                {
                    "type": "critical_alert",
                    "priority": "critical",
                    "title": f"Critical alert: {alert.message}",
                    "description": f"Alert active since {alert.timestamp.isoformat()}",
                    "recommendations": [
                        "Immediate investigation required",
                        "Check system logs",
                        "Review recent changes",
                        "Consider emergency rollback",
                    ],
                    "alert_id": str(alert.timestamp.timestamp()),
                }
            )

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        issues.sort(key=lambda x: priority_order.get(str(x["priority"]), 3))

        return issues

    async def export_metrics(self, format: str = "json") -> str:
        """Export current metrics in specified format.

        Args:
            format: Export format (json, csv)

        Returns:
            Formatted metrics data
        """
        metrics = await self.get_system_metrics()

        if format.lower() == "json":
            return json.dumps(metrics.to_dict(), indent=2)
        elif format.lower() == "csv":
            # Simple CSV export for key metrics
            lines = [
                "timestamp,parser_health_overall,cache_hit_rate,total_alerts,critical_alerts",
                f"{metrics.timestamp.isoformat()},"
                f"{sum(1 for h in metrics.parser_health.values() if h.get('healthy', False)) / len(metrics.parser_health)},"
                f"{metrics.cache_stats.get('hit_rate', 0.0)},"
                f"{sum(metrics.alert_summary.values())},"
                f"{metrics.alert_summary.get('critical', 0)}",
            ]
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    async def start_monitoring(self, interval: int = 60) -> None:
        """Start continuous monitoring with specified interval.

        Args:
            interval: Monitoring interval in seconds
        """
        print(f"Starting dashboard monitoring (interval: {interval}s)")

        while True:
            try:
                health = await self.get_health_status()
                print(
                    f"[{datetime.now().isoformat()}] System status: {health['status']} "
                    f"(score: {health['health_score']:.2f})"
                )

                if health["issues"]:
                    print("  Issues:")
                    for issue in health["issues"]:
                        print(f"    - {issue}")

                await asyncio.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(interval)


async def main() -> None:
    """Example dashboard usage."""

    # Initialize components (would use actual Redis in production)
    redis_client = None  # Redis(host="localhost", port=6379)

    alert_manager = AlertManager(redis_client=redis_client)
    structure_monitor = StructureMonitor()
    fallback_cache = FallbackCache(redis_client=redis_client)

    # Create dashboard
    dashboard = MonitoringDashboard(
        alert_manager=alert_manager,
        structure_monitor=structure_monitor,
        fallback_cache=fallback_cache,
    )

    # Get current metrics
    metrics = await dashboard.get_system_metrics()
    print("Current System Metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))

    # Get health status
    health = await dashboard.get_health_status()
    print("\nSystem Health:")
    print(json.dumps(health, indent=2))

    # Get active issues
    issues = await dashboard.get_active_issues()
    print(f"\nActive Issues ({len(issues)}):")
    for issue in issues:
        print(f"  [{issue['priority'].upper()}] {issue['title']}")
        print(f"    {issue['description']}")


if __name__ == "__main__":
    asyncio.run(main())
