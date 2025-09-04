# Documentation Templates

This directory contains standardized templates for creating consistent documentation across the Tracktion project.

## Available Templates

### Service Documentation
- **File:** `service-readme.md`
- **Purpose:** README template for microservices
- **Usage:** Copy and replace placeholders with service-specific information

### API Documentation
- **File:** `api-documentation.md`
- **Purpose:** Comprehensive API documentation template
- **Usage:** Document REST APIs, WebSocket APIs, and data models

### Tutorial Template
- **File:** `tutorial-template.md`
- **Purpose:** Step-by-step tutorial creation
- **Usage:** Create educational content with consistent structure

### Troubleshooting Template
- **File:** `troubleshooting-template.md`
- **Purpose:** Diagnostic and problem-resolution guides
- **Usage:** Document common issues and their solutions

## Using Templates

### 1. Copy Template

```bash
# Copy the appropriate template
cp docs/templates/service-readme.md services/your-service/README.md
```

### 2. Replace Placeholders

Templates use the format `{{ placeholder_name }}` for variables that need to be replaced.

Common placeholders include:
- `{{ service_name }}` - Name of the service
- `{{ service_description }}` - Brief description
- `{{ base_url }}` - API base URL
- `{{ version }}` - Current version

### 3. Customize Content

Replace placeholder sections with actual content:
- Remove unused sections
- Add service-specific sections
- Update examples with real code
- Add actual configuration values

## Template Guidelines

### Writing Style
- **Clear and Concise:** Use simple, direct language
- **Action-Oriented:** Start with verbs for instructions
- **Consistent Terminology:** Use the same terms throughout
- **User-Focused:** Write from the user's perspective

### Structure Standards
- **Logical Flow:** Information should flow logically from overview to details
- **Scannable:** Use headers, lists, and tables for easy scanning
- **Complete:** Include all necessary information for the target audience
- **Maintainable:** Keep information current and accurate

### Code Examples
- **Working Examples:** All code should be tested and functional
- **Complete Context:** Include necessary imports and setup
- **Realistic Data:** Use realistic example data, not placeholder text
- **Multiple Languages:** Provide examples in relevant languages

## Template Variables

### Common Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{ service_name }}` | Service identifier | `analysis_service` |
| `{{ service_title }}` | Human-readable name | `Analysis Service` |
| `{{ base_url }}` | API base URL | `https://api.tracktion.io/v1` |
| `{{ version }}` | Current version | `1.0.0` |
| `{{ port }}` | Default port | `8001` |

### Service-Specific Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{ db_url }}` | Database connection | `postgresql://localhost/tracktion` |
| `{{ queue_url }}` | Message queue URL | `amqp://localhost:5672` |
| `{{ cache_url }}` | Redis cache URL | `redis://localhost:6379` |

### Documentation Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{ last_updated }}` | Last update date | `2024-01-09` |
| `{{ reviewer }}` | Document reviewer | `@username` |
| `{{ difficulty_level }}` | Tutorial difficulty | `Beginner/Intermediate/Advanced` |

## Best Practices

### Before Creating Documentation

1. **Identify Audience:** Who will use this documentation?
2. **Define Purpose:** What should readers accomplish?
3. **Choose Template:** Select the most appropriate template
4. **Gather Information:** Collect all necessary details

### While Writing

1. **Follow Template Structure:** Don't skip sections unless truly unnecessary
2. **Use Consistent Formatting:** Follow markdown standards
3. **Test Examples:** Verify all code examples work
4. **Link Related Content:** Reference other relevant documentation

### After Writing

1. **Review Completeness:** Ensure all placeholders are replaced
2. **Validate Links:** Check that all links work
3. **Test Instructions:** Follow your own instructions
4. **Peer Review:** Have someone else review the content

## Template Maintenance

### Updating Templates

When updating templates:
1. Update this README with changes
2. Version templates if making breaking changes
3. Update existing documentation using the template
4. Communicate changes to the team

### Adding New Templates

To add a new template:
1. Create the template file in this directory
2. Follow existing naming conventions
3. Document the template in this README
4. Add template variables to the reference table

## Quality Checklist

Before publishing documentation, ensure:

- [ ] All placeholders replaced
- [ ] Code examples tested
- [ ] Links verified
- [ ] Spelling and grammar checked
- [ ] Consistent formatting applied
- [ ] Required sections completed
- [ ] Target audience addressed
- [ ] Peer review completed

## Support

For help with documentation templates:
- **Style Questions:** See [Style Guide](../contributing/style-guide.md)
- **Technical Issues:** Create an issue in the repository
- **Content Review:** Request review from the documentation team
