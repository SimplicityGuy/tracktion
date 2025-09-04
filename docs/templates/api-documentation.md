# {{ service_name }} API Documentation

## Overview

{{ api_overview }}

**Base URL:** `{{ base_url }}`
**Version:** `{{ api_version }}`
**Authentication:** {{ auth_type }}

## Authentication

{{ authentication_details }}

### API Key Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" {{ base_url }}/endpoint
```

### OAuth 2.0 (if applicable)

{{ oauth_details }}

## Rate Limiting

{{ rate_limiting_info }}

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "{{ error_code }}",
    "message": "{{ error_message }}",
    "details": "{{ error_details }}",
    "timestamp": "{{ timestamp }}",
    "request_id": "{{ request_id }}"
  }
}
```

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| 200  | OK | Successful request |
| 201  | Created | Resource created successfully |
| 400  | Bad Request | Invalid request parameters |
| 401  | Unauthorized | Authentication required |
| 403  | Forbidden | Access denied |
| 404  | Not Found | Resource not found |
| 422  | Unprocessable Entity | Validation error |
| 429  | Too Many Requests | Rate limit exceeded |
| 500  | Internal Server Error | Server error |

## API Endpoints

### {{ endpoint_category_1 }}

#### {{ endpoint_1_method }} {{ endpoint_1_path }}

{{ endpoint_1_description }}

**Parameters:**

| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
{{ endpoint_1_params }}

**Request Example:**

```bash
curl -X {{ endpoint_1_method }} \
  "{{ base_url }}{{ endpoint_1_path }}" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{{ endpoint_1_request_body }}'
```

**Response Example:**

```json
{{ endpoint_1_response_example }}
```

**Response Schema:**

```json
{{ endpoint_1_response_schema }}
```

---

### {{ endpoint_category_2 }}

#### {{ endpoint_2_method }} {{ endpoint_2_path }}

{{ endpoint_2_description }}

**Parameters:**

| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
{{ endpoint_2_params }}

**Request Example:**

```bash
{{ endpoint_2_request_example }}
```

**Response Example:**

```json
{{ endpoint_2_response_example }}
```

---

## WebSocket API (if applicable)

### Connection

```javascript
const ws = new WebSocket('{{ websocket_url }}');
```

### Message Format

```json
{
  "type": "{{ message_type }}",
  "data": {{ message_data }},
  "timestamp": "{{ timestamp }}"
}
```

### Event Types

| Event Type | Description | Data Format |
|------------|-------------|-------------|
{{ websocket_events }}

## Data Models

### {{ model_1_name }}

```json
{{ model_1_schema }}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
{{ model_1_fields }}

### {{ model_2_name }}

```json
{{ model_2_schema }}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
{{ model_2_fields }}

## Examples

### Example 1: {{ example_1_title }}

{{ example_1_description }}

```bash
{{ example_1_code }}
```

### Example 2: {{ example_2_title }}

{{ example_2_description }}

```python
{{ example_2_code }}
```

### Example 3: {{ example_3_title }}

{{ example_3_description }}

```javascript
{{ example_3_code }}
```

## SDKs and Libraries

### Python SDK

```bash
pip install {{ python_sdk_name }}
```

```python
{{ python_sdk_example }}
```

### JavaScript SDK

```bash
npm install {{ js_sdk_name }}
```

```javascript
{{ js_sdk_example }}
```

## Postman Collection

{{ postman_collection_info }}

## Testing

### Test Environment

**Base URL:** `{{ test_base_url }}`
**Test API Key:** `{{ test_api_key }}`

### Sample Test Cases

{{ test_cases }}

## Changelog

### Version {{ current_version }}

{{ version_changes }}

### Version {{ previous_version }}

{{ previous_version_changes }}

## Support

- **API Issues**: {{ api_issues_url }}
- **Documentation**: {{ api_docs_url }}
- **Status Page**: {{ status_page_url }}
- **Developer Forum**: {{ forum_url }}
