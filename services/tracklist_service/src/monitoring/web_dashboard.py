"""Web-based monitoring dashboard for resilient scraping system."""

import aiohttp_cors
from aiohttp import web
from aiohttp.web import Application, Request, Response
from services.tracklist_service.src.cache.fallback_cache import FallbackCache
from services.tracklist_service.src.monitoring.alert_manager import AlertManager
from services.tracklist_service.src.monitoring.dashboard import MonitoringDashboard
from services.tracklist_service.src.monitoring.structure_monitor import StructureMonitor


class WebMonitoringDashboard:
    """Web-based monitoring dashboard with REST API and HTML interface."""

    def __init__(self, dashboard: MonitoringDashboard):
        """Initialize web dashboard.

        Args:
            dashboard: Core monitoring dashboard
        """
        self.dashboard = dashboard
        self.app = self._create_app()

    def _create_app(self) -> Application:
        """Create aiohttp application with routes."""
        app = web.Application()

        # Configure CORS
        cors = aiohttp_cors.setup(
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        # API routes
        app.router.add_get("/", self.index)
        app.router.add_get("/api/health", self.api_health)
        app.router.add_get("/api/metrics", self.api_metrics)
        app.router.add_get("/api/issues", self.api_issues)
        app.router.add_get("/api/trends", self.api_trends)
        app.router.add_get("/api/export", self.api_export)

        # Add CORS to all routes
        for route in app.router.routes():
            cors.add(route)

        return app

    async def index(self, request: Request) -> Response:
        """Serve dashboard HTML page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Resilient Scraping System Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
        .metric-value { font-size: 24px; font-weight: bold; margin-bottom: 5px; }
        .metric-description { color: #666; font-size: 14px; }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .issues-section { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .issue-item { border-left: 4px solid #3498db; padding: 10px; margin-bottom: 10px; background: #f8f9fa; }
        .issue-critical { border-left-color: #e74c3c; }
        .issue-high { border-left-color: #f39c12; }
        .issue-medium { border-left-color: #3498db; }
        .refresh-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        .refresh-btn:hover { background: #2980b9; }
        .timestamp { color: #666; font-size: 12px; }
        #loading { display: none; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Resilient Scraping System Dashboard</h1>
            <p>Real-time monitoring and alerting for web scraping infrastructure</p>
            <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
            <span id="loading">Loading...</span>
            <div class="timestamp" id="lastUpdate"></div>
        </div>

        <div class="metrics-grid" id="metricsGrid">
            <!-- Metrics will be loaded here -->
        </div>

        <div class="issues-section">
            <h2>Active Issues</h2>
            <div id="issuesList">
                <!-- Issues will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        async function fetchData(endpoint) {
            const response = await fetch(endpoint);
            return await response.json();
        }

        async function refreshData() {
            document.getElementById('loading').style.display = 'inline';

            try {
                // Fetch health status
                const health = await fetchData('/api/health');

                // Fetch detailed metrics
                const metrics = await fetchData('/api/metrics');

                // Fetch active issues
                const issues = await fetchData('/api/issues');

                // Update UI
                updateMetrics(health, metrics);
                updateIssues(issues);

                document.getElementById('lastUpdate').textContent =
                    `Last updated: ${new Date().toLocaleString()}`;

            } catch (error) {
                console.error('Error fetching data:', error);
                alert('Error fetching dashboard data');
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        function updateMetrics(health, metrics) {
            const grid = document.getElementById('metricsGrid');

            const statusClass = `status-${health.status}`;

            grid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-title">System Health</div>
                    <div class="metric-value ${statusClass}">${health.status.toUpperCase()}</div>
                    <div class="metric-description">Score: ${(health.health_score * 100).toFixed(1)}%</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Parser Health</div>
                    <div class="metric-value">
                        ${health.uptime_metrics.parsers_healthy}/${health.uptime_metrics.total_parsers}
                    </div>
                    <div class="metric-description">Healthy parsers</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Cache Performance</div>
                    <div class="metric-value">${(health.uptime_metrics.cache_hit_rate * 100).toFixed(1)}%</div>
                    <div class="metric-description">Hit rate</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Alerts (24h)</div>
                    <div class="metric-value">${health.uptime_metrics.alerts_24h}</div>
                    <div class="metric-description">Total alerts</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Cache Statistics</div>
                    <div class="metric-value">${metrics.cache_stats.hits + metrics.cache_stats.misses}</div>
                    <div class="metric-description">
                        Hits: ${metrics.cache_stats.hits} |
                        Misses: ${metrics.cache_stats.misses} |
                        Fallback: ${metrics.cache_stats.fallback_hits}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Alert Breakdown</div>
                    <div class="metric-value">${Object.values(metrics.alert_summary).reduce((a, b) => a + b, 0)}</div>
                    <div class="metric-description">
                        Critical: ${metrics.alert_summary.critical || 0} |
                        Error: ${metrics.alert_summary.error || 0} |
                        Warning: ${metrics.alert_summary.warning || 0}
                    </div>
                </div>
            `;
        }

        function updateIssues(issues) {
            const list = document.getElementById('issuesList');

            if (issues.length === 0) {
                list.innerHTML = '<p style="color: #27ae60;">No active issues detected.</p>';
                return;
            }

            list.innerHTML = issues.map(issue => `
                <div class="issue-item issue-${issue.priority}">
                    <strong>[${issue.priority.toUpperCase()}] ${issue.title}</strong>
                    <p>${issue.description}</p>
                    <details>
                        <summary>Recommendations</summary>
                        <ul>
                            ${issue.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </details>
                </div>
            `).join('');
        }

        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);

        // Initial load
        refreshData();
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type="text/html")

    async def api_health(self, request: Request) -> Response:
        """API endpoint for system health status."""
        try:
            health = await self.dashboard.get_health_status()
            return web.json_response(health)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def api_metrics(self, request: Request) -> Response:
        """API endpoint for system metrics."""
        try:
            page_types_param = request.query.get("page_types")
            page_types: list[str] | None = None
            if page_types_param:
                page_types = page_types_param.split(",")

            metrics = await self.dashboard.get_system_metrics(page_types)
            return web.json_response(metrics.to_dict())
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def api_issues(self, request: Request) -> Response:
        """API endpoint for active issues."""
        try:
            issues = await self.dashboard.get_active_issues()
            return web.json_response(issues)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def api_trends(self, request: Request) -> Response:
        """API endpoint for performance trends."""
        try:
            hours = int(request.query.get("hours", 24))
            trends = await self.dashboard.get_performance_trends(hours)
            return web.json_response(trends)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def api_export(self, request: Request) -> Response:
        """API endpoint for metrics export."""
        try:
            format_type = request.query.get("format", "json")
            data = await self.dashboard.export_metrics(format_type)

            if format_type.lower() == "csv":
                return web.Response(
                    text=data,
                    content_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=metrics.csv"},
                )
            return web.Response(
                text=data,
                content_type="application/json",
                headers={"Content-Disposition": "attachment; filename=metrics.json"},
            )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def start_server(self, host: str = "localhost", port: int = 8080) -> None:
        """Start the web server.

        Args:
            host: Server host
            port: Server port
        """
        print(f"Starting web dashboard at http://{host}:{port}")

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        print(f"Dashboard available at http://{host}:{port}")
        print("API endpoints:")
        print(f"  - GET http://{host}:{port}/api/health")
        print(f"  - GET http://{host}:{port}/api/metrics")
        print(f"  - GET http://{host}:{port}/api/issues")
        print(f"  - GET http://{host}:{port}/api/trends?hours=24")
        print(f"  - GET http://{host}:{port}/api/export?format=json|csv")


async def create_web_dashboard() -> WebMonitoringDashboard:
    """Create web dashboard with default components."""

    # Initialize components
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

    return WebMonitoringDashboard(dashboard)


async def main() -> None:
    """Start web dashboard server."""
    web_dashboard = await create_web_dashboard()
    await web_dashboard.start_server(host="0.0.0.0", port=8080)

    # Keep server running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down dashboard...")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
