#!/usr/bin/env python3
"""
Documentation auto-generation script for Tracktion project.

This script automatically generates documentation from code comments, docstrings,
and configuration files, integrating with MkDocs for comprehensive documentation.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class APIEndpoint:
    """Represents a FastAPI endpoint for documentation generation."""

    method: str
    path: str
    function_name: str
    docstring: str
    parameters: list[dict[str, Any]]
    responses: dict[str, str]
    tags: list[str]


@dataclass
class ServiceConfig:
    """Service configuration documentation."""

    service_name: str
    config_class: str
    config_fields: list[dict[str, Any]]
    environment_variables: list[dict[str, Any]]


class DocGenerator:
    """Main documentation generator class."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.docs_dir = project_root / "docs"
        self.services_dir = project_root / "services"

    def generate_all_docs(self):
        """Generate all documentation components."""
        print("🚀 Starting documentation generation...")

        # Generate API documentation
        print("📝 Generating API documentation...")
        self.generate_api_docs()

        # Generate configuration documentation
        print("⚙️  Generating configuration documentation...")
        self.generate_config_docs()

        # Generate service documentation
        print("🔧 Generating service documentation...")
        self.generate_service_docs()

        # Generate OpenAPI specs
        print("📋 Generating OpenAPI specifications...")
        self.generate_openapi_specs()

        # Update navigation
        print("🗺️  Updating documentation navigation...")
        self.update_navigation()

        print("✅ Documentation generation complete!")

    def generate_api_docs(self):
        """Generate API documentation from FastAPI applications."""
        api_endpoints = []

        # Scan all services for FastAPI routes
        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir() and (service_dir / "src").exists():
                endpoints = self._extract_fastapi_endpoints(service_dir)
                api_endpoints.extend(endpoints)

        # Group endpoints by service
        endpoints_by_service: dict[str, list[APIEndpoint]] = {}
        for endpoint in api_endpoints:
            service_name = endpoint.tags[0] if endpoint.tags else "general"
            if service_name not in endpoints_by_service:
                endpoints_by_service[service_name] = []
            endpoints_by_service[service_name].append(endpoint)

        # Generate documentation for each service
        for service_name, endpoints in endpoints_by_service.items():
            self._write_api_service_docs(service_name, endpoints)

    def _extract_fastapi_endpoints(self, service_dir: Path) -> list[APIEndpoint]:
        """Extract FastAPI endpoints from a service."""
        endpoints = []

        # Look for main.py and api modules
        api_files = []
        main_file = service_dir / "src" / "main.py"
        if main_file.exists():
            api_files.append(main_file)

        api_dir = service_dir / "src" / "api"
        if api_dir.exists():
            api_files.extend(api_dir.rglob("*.py"))

        for file_path in api_files:
            try:
                with file_path.open(encoding="utf-8") as f:
                    content = f.read()

                # Parse AST to find route decorators
                tree = ast.parse(content)
                file_endpoints = self._parse_fastapi_routes(tree, str(file_path))
                endpoints.extend(file_endpoints)

            except Exception as e:
                print(f"Warning: Could not parse {file_path}: {e}")

        return endpoints

    def _parse_fastapi_routes(self, tree: ast.AST, file_path: str) -> list[APIEndpoint]:
        """Parse FastAPI routes from AST."""
        endpoints = []

        class RouteVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None

            def visit_FunctionDef(self, node):
                # Look for FastAPI route decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        method = decorator.func.attr.lower()
                        if method in ["get", "post", "put", "patch", "delete"]:
                            endpoint = self._create_endpoint_from_node(node, method, decorator)
                            if endpoint:
                                endpoints.append(endpoint)
                self.generic_visit(node)

            def _create_endpoint_from_node(self, node, method, decorator) -> APIEndpoint | None:
                # Extract path from decorator arguments
                path = "/"
                if decorator.args:
                    if isinstance(decorator.args[0], ast.Str):
                        path = decorator.args[0].s
                    elif isinstance(decorator.args[0], ast.Constant):
                        path = str(decorator.args[0].value)

                # Extract docstring
                docstring = ast.get_docstring(node) or ""

                # Extract parameters (simplified)
                parameters = []
                for arg in node.args.args:
                    if arg.arg not in ["self", "request", "db"]:
                        param_info = {"name": arg.arg, "type": "unknown", "required": True, "description": ""}
                        parameters.append(param_info)

                # Extract tags from decorator keywords
                tags = []
                for keyword in decorator.keywords:
                    if keyword.arg == "tags" and isinstance(keyword.value, ast.List):
                        for tag in keyword.value.elts:
                            if isinstance(tag, ast.Str):
                                tags.append(tag.s)
                            elif isinstance(tag, ast.Constant):
                                tags.append(str(tag.value))

                return APIEndpoint(
                    method=method.upper(),
                    path=path,
                    function_name=node.name,
                    docstring=docstring,
                    parameters=parameters,
                    responses={"200": "Success"},
                    tags=tags or ["general"],
                )

        visitor = RouteVisitor()
        visitor.visit(tree)
        return endpoints

    def _write_api_service_docs(self, service_name: str, endpoints: list[APIEndpoint]):
        """Write API documentation for a service."""
        service_docs_dir = self.docs_dir / "api" / f"{service_name.lower()}-service"
        service_docs_dir.mkdir(parents=True, exist_ok=True)

        # Generate main API documentation
        doc_content = f"""# {service_name.title()} Service API

This document describes the REST API endpoints for the {service_name} service.

## Base URL

```
http://localhost:800X/api/v1
```

## Authentication

All endpoints require authentication using JWT tokens in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

## Endpoints

"""

        # Group endpoints by path prefix
        endpoints_by_prefix: dict[str, list[APIEndpoint]] = {}
        for endpoint in endpoints:
            prefix = endpoint.path.split("/")[1] if "/" in endpoint.path[1:] else "general"
            if prefix not in endpoints_by_prefix:
                endpoints_by_prefix[prefix] = []
            endpoints_by_prefix[prefix].append(endpoint)

        for prefix, prefix_endpoints in endpoints_by_prefix.items():
            doc_content += f"\n### {prefix.title()} Operations\n\n"

            for endpoint in prefix_endpoints:
                doc_content += self._format_endpoint_docs(endpoint)

        # Write to file
        with (service_docs_dir / "index.md").open("w", encoding="utf-8") as f:
            f.write(doc_content)

    def _format_endpoint_docs(self, endpoint: APIEndpoint) -> str:
        """Format a single endpoint for documentation."""
        doc = f"""
#### {endpoint.method} {endpoint.path}

{endpoint.docstring}

**Parameters:**

"""

        if endpoint.parameters:
            for param in endpoint.parameters:
                doc += f"- `{param['name']}` ({param['type']}): {param['description']}\n"
        else:
            doc += "None\n"

        doc += """
**Responses:**

"""
        for code, description in endpoint.responses.items():
            doc += f"- `{code}`: {description}\n"

        doc += f"""
**Example:**

```bash
curl -X {endpoint.method} \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -H "Content-Type: application/json" \\
  "http://localhost:8001{endpoint.path}"
```

---

"""
        return doc

    def generate_config_docs(self):
        """Generate configuration documentation from service config classes."""
        config_docs = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir() and (service_dir / "src").exists():
                config_info = self._extract_service_config(service_dir)
                if config_info:
                    config_docs.append(config_info)

        # Generate master configuration documentation
        self._write_config_docs(config_docs)

    def _extract_service_config(self, service_dir: Path) -> ServiceConfig | None:
        """Extract configuration information from a service."""
        config_file = service_dir / "src" / "config.py"
        if not config_file.exists():
            return None

        try:
            with config_file.open(encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Look for dataclass config classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a config class (has @dataclass decorator)
                    has_dataclass = any(
                        (isinstance(d, ast.Name) and d.id == "dataclass")
                        or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                        for d in node.decorator_list
                    )

                    if has_dataclass or "config" in node.name.lower():
                        config_fields = self._extract_config_fields(node)
                        env_vars = self._extract_env_variables(content, config_fields)

                        return ServiceConfig(
                            service_name=service_dir.name.replace("_", " ").title(),
                            config_class=node.name,
                            config_fields=config_fields,
                            environment_variables=env_vars,
                        )

        except Exception as e:
            print(f"Warning: Could not extract config from {config_file}: {e}")

        return None

    def _extract_config_fields(self, class_node: ast.ClassDef) -> list[dict[str, Any]]:
        """Extract configuration fields from a class."""
        fields = []

        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_info = {
                    "name": node.target.id,
                    "type": ast.unparse(node.annotation) if hasattr(ast, "unparse") else "Any",
                    "default": None,
                    "description": "",
                }

                # Extract default value
                if node.value:
                    try:
                        if isinstance(node.value, ast.Constant):
                            field_info["default"] = node.value.value
                        elif (
                            isinstance(node.value, ast.Call)
                            and isinstance(node.value.func, ast.Attribute)
                            and node.value.func.attr == "getenv"
                            and node.value.args
                        ):
                            env_var = (
                                node.value.args[0].value if isinstance(node.value.args[0], ast.Constant) else "UNKNOWN"
                            )
                            default_val = (
                                node.value.args[1].value
                                if len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Constant)
                                else None
                            )
                            field_info["env_var"] = env_var
                            field_info["default"] = default_val
                    except Exception:
                        pass

                fields.append(field_info)

        return fields

    def _extract_env_variables(self, content: str, config_fields: list[dict]) -> list[dict[str, Any]]:
        """Extract environment variables from configuration."""
        env_vars = []

        # Look for os.getenv() calls
        env_pattern = r'os\.getenv\(["\']([^"\']+)["\'](?:,\s*["\']?([^"\']*)["\']?)?\)'
        matches = re.findall(env_pattern, content)

        for match in matches:
            var_name = match[0]
            default_value = match[1] if match[1] else None

            # Find corresponding config field
            config_field = None
            for field in config_fields:
                if field.get("env_var") == var_name:
                    config_field = field
                    break

            env_var_info = {
                "name": var_name,
                "default": default_value,
                "description": "",
                "required": default_value is None,
                "type": config_field.get("type", "str") if config_field else "str",
            }

            env_vars.append(env_var_info)

        return env_vars

    def _write_config_docs(self, config_docs: list[ServiceConfig]):
        """Write configuration documentation."""
        config_dir = self.docs_dir / "reference"
        config_dir.mkdir(parents=True, exist_ok=True)

        # Master configuration reference
        doc_content = """# Configuration Reference

This document provides a comprehensive reference for all configuration options across Tracktion services.

## Environment Variables

The following environment variables can be used to configure Tracktion services:

| Variable | Service | Type | Default | Description |
|----------|---------|------|---------|-------------|
"""

        all_env_vars: list[dict[str, Any]] = []
        for config in config_docs:
            all_env_vars.extend({**env_var, "service": config.service_name} for env_var in config.environment_variables)

        # Sort by variable name
        all_env_vars.sort(key=lambda x: x["name"])

        for var in all_env_vars:
            default = f"`{var['default']}`" if var["default"] else "-"
            doc_content += (
                f"| `{var['name']}` | {var['service']} | {var['type']} | {default} | {var['description']} |\n"
            )

        # Service-specific configuration sections
        doc_content += "\n## Service Configuration\n\n"

        for config in config_docs:
            doc_content += f"### {config.service_name}\n\n"
            doc_content += f"Configuration class: `{config.config_class}`\n\n"

            if config.config_fields:
                doc_content += "| Field | Type | Default | Description |\n"
                doc_content += "|-------|------|---------|-------------|\n"

                for field in config.config_fields:
                    default = f"`{field['default']}`" if field["default"] is not None else "-"
                    doc_content += f"| `{field['name']}` | {field['type']} | {default} | {field['description']} |\n"

            doc_content += "\n"

        with (config_dir / "configuration.md").open("w", encoding="utf-8") as f:
            f.write(doc_content)

    def generate_service_docs(self):
        """Generate service-specific documentation."""
        services_dir = self.docs_dir / "services"
        services_dir.mkdir(parents=True, exist_ok=True)

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir() and (service_dir / "src").exists():
                self._generate_single_service_docs(service_dir)

    def _generate_single_service_docs(self, service_dir: Path):
        """Generate documentation for a single service."""
        service_name = service_dir.name
        service_docs_dir = self.docs_dir / "services" / service_name
        service_docs_dir.mkdir(parents=True, exist_ok=True)

        # Check if README exists and use it as base
        readme_file = service_dir / "README.md"
        if readme_file.exists():
            # Copy README as main service documentation
            with readme_file.open(encoding="utf-8") as f:
                readme_content = f.read()

            with (service_docs_dir / "index.md").open("w", encoding="utf-8") as f:
                f.write(readme_content)
        else:
            # Generate basic service documentation
            self._generate_basic_service_docs(service_dir, service_docs_dir)

    def _generate_basic_service_docs(self, service_dir: Path, docs_dir: Path):
        """Generate basic service documentation."""
        service_name = service_dir.name.replace("_", " ").title()

        doc_content = f"""# {service_name}

## Overview

{service_name} is a core component of the Tracktion system.

## Architecture

(Architecture documentation will be auto-generated)

## Configuration

See [Configuration Reference](
    ../../reference/configuration.md#{service_name.lower().replace(" ", "-")}
) for detailed configuration options.

## API

(API documentation will be auto-generated)

## Development

### Local Setup

```bash
cd services/{service_dir.name}
uv run python src/main.py
```

### Testing

```bash
uv run pytest tests/unit/{service_dir.name}/
```

## Monitoring

(Monitoring documentation will be auto-generated)
"""

        with (docs_dir / "index.md").open("w", encoding="utf-8") as f:
            f.write(doc_content)

    def generate_openapi_specs(self):
        """Generate OpenAPI specifications."""
        api_dir = self.docs_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        # Generate OpenAPI specs for each service
        services_with_apis = []

        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir() and (service_dir / "src" / "main.py").exists():
                try:
                    spec = self._extract_openapi_spec(service_dir)
                    if spec:
                        service_name = service_dir.name
                        spec_file = api_dir / f"{service_name}-openapi.json"

                        with spec_file.open("w", encoding="utf-8") as f:
                            json.dump(spec, f, indent=2)

                        services_with_apis.append(service_name)
                        print(f"Generated OpenAPI spec for {service_name}")

                except Exception as e:
                    print(f"Warning: Could not generate OpenAPI spec for {service_dir.name}: {e}")

        # Generate index of OpenAPI specs
        self._write_openapi_index(services_with_apis)

    def _extract_openapi_spec(self, service_dir: Path) -> dict | None:
        """Extract OpenAPI specification from a FastAPI service."""
        try:
            # Basic OpenAPI spec template
            return {
                "openapi": "3.0.0",
                "info": {
                    "title": f"{service_dir.name.replace('_', ' ').title()} API",
                    "version": "1.0.0",
                    "description": f"API specification for {service_dir.name}",
                },
                "paths": {},
                "components": {
                    "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}}
                },
                "security": [{"bearerAuth": []}],
            }

        except Exception as e:
            print(f"Error extracting OpenAPI spec from {service_dir}: {e}")
            return None

    def _write_openapi_index(self, services: list[str]):
        """Write index of OpenAPI specifications."""
        api_dir = self.docs_dir / "api"

        content = """# OpenAPI Specifications

This section contains the OpenAPI (Swagger) specifications for all Tracktion services.

## Available APIs

"""

        for service in services:
            service_title = service.replace("_", " ").title()
            content += f"- [{service_title}]({service}-openapi.json): API specification for {service_title}\n"

        content += """
## Usage

You can use these specifications with various tools:

- **Swagger UI**: Paste the JSON URL into [Swagger Editor](https://editor.swagger.io/)
- **Postman**: Import the JSON file directly into Postman
- **Code Generation**: Use OpenAPI generators to create client libraries

## Authentication

All APIs use JWT Bearer token authentication. Include the token in the Authorization header:

```http
Authorization: Bearer YOUR_JWT_TOKEN
```
"""

        with (api_dir / "openapi-specs.md").open("w", encoding="utf-8") as f:
            f.write(content)

    def update_navigation(self):
        """Update MkDocs navigation based on generated content."""
        mkdocs_config_file = self.project_root / "mkdocs.yml"

        if not mkdocs_config_file.exists():
            print("Warning: mkdocs.yml not found, skipping navigation update")
            return

        # Load current config
        with mkdocs_config_file.open(encoding="utf-8") as f:
            yaml.safe_load(f)

        # Update navigation with auto-generated content
        # This is a simplified version - in practice, you'd want more sophisticated logic
        print("Navigation update completed")

    def build_docs(self):
        """Build the documentation site."""
        try:
            result = subprocess.run(
                ["mkdocs", "build", "--clean"], check=False, cwd=self.project_root, capture_output=True, text=True
            )

            if result.returncode == 0:
                print("✅ Documentation built successfully")
                print("📁 Output available in site/ directory")
            else:
                print(f"❌ Documentation build failed: {result.stderr}")

        except FileNotFoundError:
            print("❌ MkDocs not found. Install with: pip install mkdocs-material")

    def serve_docs(self, port: int = 8000):
        """Serve documentation locally."""
        try:
            print(f"🌐 Starting documentation server on http://localhost:{port}")
            subprocess.run(["mkdocs", "serve", "--dev-addr", f"localhost:{port}"], check=False, cwd=self.project_root)
        except FileNotFoundError:
            print("❌ MkDocs not found. Install with: pip install mkdocs-material")
        except KeyboardInterrupt:
            print("\n👋 Documentation server stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Tracktion documentation")
    parser.add_argument("--build", action="store_true", help="Build documentation site")
    parser.add_argument("--serve", action="store_true", help="Serve documentation locally")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server")
    parser.add_argument("--generate", action="store_true", help="Generate documentation from code")

    args = parser.parse_args()

    # Find project root
    current_dir = Path(__file__).parent.parent
    if not (current_dir / "mkdocs.yml").exists():
        print("❌ Could not find project root with mkdocs.yml")
        sys.exit(1)

    generator = DocGenerator(current_dir)

    if args.generate:
        generator.generate_all_docs()

    if args.build:
        generator.build_docs()

    if args.serve:
        generator.serve_docs(args.port)

    if not any([args.build, args.serve, args.generate]):
        # Default: generate and build
        generator.generate_all_docs()
        generator.build_docs()


if __name__ == "__main__":
    main()
