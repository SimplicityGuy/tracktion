# {{ service_name }}

## Overview

{{ service_description }}

## Features

{{ service_features }}

## Architecture

### Service Design

{{ architecture_overview }}

### Components

{{ component_list }}

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
{{ dependency_table }}

## Installation

### Prerequisites

{{ prerequisites }}

### Development Setup

```bash
# Clone repository
git clone {{ repo_url }}
cd {{ service_directory }}

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run database migrations (if applicable)
{{ migration_commands }}

# Start the service
{{ start_commands }}
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
{{ env_vars_table }}

### Configuration Files

{{ config_files_description }}

## Usage

### Basic Usage

{{ basic_usage_examples }}

### Advanced Usage

{{ advanced_usage_examples }}

## API Reference

{{ api_reference_link }}

### Key Endpoints

{{ key_endpoints_list }}

## Testing

### Running Tests

```bash
# Unit tests
uv run pytest tests/unit/

# Integration tests
uv run pytest tests/integration/

# All tests
uv run pytest

# Coverage report
uv run pytest --cov={{ service_name }} --cov-report=html
```

### Test Structure

{{ test_structure }}

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t {{ service_name }}:{{ version }} .

# Run container
docker run -d \
  --name {{ service_name }} \
  -p {{ port }}:{{ port }} \
  -e DATABASE_URL={{ db_url }} \
  {{ service_name }}:{{ version }}
```

### Kubernetes Deployment

{{ k8s_deployment_info }}

## Monitoring

### Health Checks

{{ health_check_endpoints }}

### Metrics

{{ metrics_description }}

### Logging

{{ logging_configuration }}

## Troubleshooting

### Common Issues

#### Issue: {{ common_issue_1 }}
**Symptoms:** {{ symptoms_1 }}
**Solution:** {{ solution_1 }}

#### Issue: {{ common_issue_2 }}
**Symptoms:** {{ symptoms_2 }}
**Solution:** {{ solution_2 }}

### Debugging

{{ debugging_steps }}

### Performance Tuning

{{ performance_tips }}

## Contributing

See [Contributing Guide](../contributing/guidelines.md) for development guidelines.

### Development Workflow

{{ dev_workflow }}

## Security

{{ security_considerations }}

## Performance

### Benchmarks

{{ performance_benchmarks }}

### Optimization Tips

{{ optimization_tips }}

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history.

## License

{{ license_info }}

## Support

- **Documentation**: {{ docs_url }}
- **Issues**: {{ issues_url }}
- **Discussions**: {{ discussions_url }}
- **Slack**: {{ slack_channel }}
