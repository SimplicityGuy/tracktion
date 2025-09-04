# Documentation Style Guide

This guide defines the writing standards and conventions for all Tracktion documentation.

## Purpose and Scope

This style guide ensures consistent, clear, and professional documentation across the Tracktion project. It covers:

- Writing style and tone
- Document structure and formatting
- Technical content guidelines
- Visual presentation standards
- Accessibility requirements

## Writing Principles

### 1. Clarity First
- **Be Direct**: Use simple, clear sentences
- **Avoid Jargon**: Explain technical terms when first used
- **Use Active Voice**: "The service processes requests" (not "Requests are processed by the service")
- **Be Specific**: Use concrete examples and precise language

### 2. User-Focused Content
- **Know Your Audience**: Write for the intended user level (beginner, intermediate, advanced)
- **Answer Questions**: Address what users want to accomplish
- **Provide Context**: Explain why something matters, not just how to do it
- **Include Examples**: Show real-world usage

### 3. Consistency
- **Follow Templates**: Use established document templates
- **Maintain Terminology**: Use consistent technical terms throughout
- **Apply Formatting Rules**: Follow formatting standards uniformly
- **Use Standard Structure**: Organize content predictably

## Writing Style

### Tone and Voice

**Professional but Approachable**
- Be helpful and supportive
- Avoid overly casual language
- Use encouraging language for complex topics
- Be respectful of user expertise levels

**Examples:**
```markdown
✅ Good: "This configuration option controls the number of worker processes."
❌ Avoid: "This thing lets you mess with how many workers you want."

✅ Good: "If you encounter this error, check your database connection."
❌ Avoid: "When this blows up, make sure your DB isn't broken."
```

### Grammar and Usage

**Sentence Structure**
- Use parallel structure in lists
- Keep sentences under 25 words when possible
- Use bullet points for complex lists
- Break up long paragraphs (max 4-5 sentences)

**Punctuation**
- Use serial commas (Oxford commas)
- Avoid excessive exclamation points
- Use colons to introduce lists or explanations
- Use em dashes sparingly for emphasis

**Capitalization**
- Use sentence case for headings
- Capitalize proper nouns (service names, technologies)
- Don't capitalize common technical terms (API, database, server)

## Document Structure

### Standard Document Elements

**Every document should include:**

1. **Title**: Clear, descriptive H1 heading
2. **Overview**: Brief description of purpose and scope
3. **Table of Contents**: For documents over 500 words (auto-generated)
4. **Main Content**: Structured with logical headings
5. **Conclusion/Next Steps**: Clear action items or follow-up
6. **Metadata**: Last updated date, reviewer, status

### Heading Hierarchy

```markdown
# Page Title (H1) - One per document
## Major Section (H2)
### Subsection (H3)
#### Detail Section (H4)
##### Minor Detail (H5) - Use sparingly
```

**Heading Guidelines:**
- Use descriptive, action-oriented headings
- Don't skip heading levels
- Keep headings under 8 words
- Use parallel structure in same-level headings

### Content Organization

**Information Architecture:**
1. **Overview** → **Details** → **Examples** → **Reference**
2. **Problem** → **Solution** → **Implementation** → **Validation**
3. **What** → **Why** → **How** → **When**

## Technical Content Guidelines

### Code Examples

**All code examples must be:**
- **Functional**: Actually work as written
- **Complete**: Include necessary imports and setup
- **Realistic**: Use meaningful variable names and realistic data
- **Tested**: Verified to work with current versions

```markdown
# Good code example
```python
from tracktion.analysis import BPMDetector

# Initialize BPM detector with default settings
detector = BPMDetector(confidence_threshold=0.8)

# Analyze audio file
result = detector.analyze("path/to/audio.mp3")
print(f"Detected BPM: {result.bpm} (confidence: {result.confidence})")
```

# Bad code example
```python
# This is bad - incomplete and unclear
detector = Thing()
result = detector.do_stuff(file)
print(result)
```
```

### API Documentation

**Endpoint Documentation Format:**
```markdown
#### POST /api/v1/analyze

Analyzes an audio file and returns BPM, key, and mood information.

**Parameters:**

| Parameter | Type | Location | Required | Description |
|-----------|------|----------|----------|-------------|
| `file_path` | string | body | Yes | Path to audio file |
| `analysis_type` | string | body | No | Type of analysis (default: "full") |

**Request Example:**
```bash
curl -X POST "https://api.tracktion.io/v1/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/audio.mp3",
    "analysis_type": "full"
  }'
```

**Response Example:**
```json
{
  "bpm": 128.5,
  "key": "C major",
  "mood": "energetic",
  "confidence": 0.92
}
```
```

### Configuration Documentation

**Environment Variables:**
```markdown
### Environment Variables

| Variable | Description | Default | Required | Example |
|----------|-------------|---------|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes | `postgresql://user:pass@localhost/tracktion` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` | No | `redis://redis:6379/0` |
| `LOG_LEVEL` | Logging verbosity | `INFO` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
```

## Formatting Standards

### Markdown Conventions

**Text Formatting:**
- **Bold** for UI elements, important terms, and emphasis
- *Italic* for filenames, variable names, and new concepts
- `Code` for commands, functions, and technical terms
- Use blockquotes (>) for important notes or warnings

**Lists:**
- Use `-` for unordered lists
- Use `1.` for ordered lists
- Indent sublists with 2 spaces
- Keep list items parallel in structure

**Links:**
- Use descriptive link text (not "click here")
- Link to relevant internal documentation
- Verify external links are current
- Use reference-style links for repeated URLs

### Tables

**Table Guidelines:**
- Include descriptive headers
- Keep cell content concise
- Use tables for structured data comparison
- Ensure tables are responsive (don't exceed page width)
- Always include a table caption if context isn't clear

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |
| Data D   | Data E   | Data F   |
```

### Code Blocks

**Code Block Standards:**
```markdown
# Always specify language
```python
def example_function():
    return "Always use syntax highlighting"
```

# Include comments for complex code
```bash
# Start the development server
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

# Use realistic examples
```yaml
# docker-compose.yml
version: '3.8'
services:
  analysis:
    build: ./services/analysis_service
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/tracktion
```
```

## Visual Elements

### Admonitions

Use admonitions to highlight important information:

```markdown
!!! note "Information"
    Additional context that's helpful but not critical.

!!! tip "Pro Tip"
    Helpful suggestions or best practices.

!!! warning "Important"
    Critical information that could cause issues if ignored.

!!! danger "Caution"
    Information about potentially destructive actions.
```

### Diagrams and Images

**Image Guidelines:**
- Use descriptive alt text for accessibility
- Include captions for complex images
- Optimize images for web (WebP/PNG, <500KB)
- Use consistent styling and colors
- Ensure images are legible at different screen sizes

**Diagram Standards:**
- Use Mermaid for simple flow charts and architecture diagrams
- Use consistent colors and styling
- Include descriptive titles and labels
- Ensure diagrams are accessible (alt text, high contrast)

```markdown
```mermaid
graph TD
    A[User Request] --> B[API Gateway]
    B --> C[Analysis Service]
    C --> D[Response]
```
```

## Language and Terminology

### Standard Terms

**Use These Terms Consistently:**
- **Service** (not microservice, component, app)
- **API endpoint** (not route, handler, function)
- **Configuration** (not config, setup, settings)
- **Database** (not DB, data store)
- **Authentication** (not auth, login)
- **Authorization** (not authz, permissions)

### Technical Accuracy

**Version-Specific Information:**
- Always specify version requirements
- Test examples with current versions
- Update deprecated information promptly
- Note compatibility requirements

**Error Messages:**
- Include exact error text
- Provide context for when errors occur
- Offer clear resolution steps
- Link to related troubleshooting sections

## Accessibility Guidelines

### Content Accessibility

**Structure:**
- Use semantic heading structure (H1 → H2 → H3)
- Provide descriptive link text
- Include alt text for all images
- Use high contrast colors in diagrams

**Language:**
- Define acronyms on first use
- Avoid unnecessarily complex vocabulary
- Use clear, logical organization
- Provide context for technical concepts

### Screen Reader Compatibility

**Tables:**
- Use proper table headers
- Include table captions when needed
- Avoid complex table layouts

**Code Examples:**
- Include descriptive comments
- Use semantic markup for code
- Provide context before code blocks

## Review and Maintenance

### Quality Checklist

Before publishing documentation:

- [ ] Content is accurate and current
- [ ] Examples have been tested
- [ ] Links work correctly
- [ ] Spelling and grammar checked
- [ ] Formatting is consistent
- [ ] Accessibility requirements met
- [ ] Technical review completed
- [ ] User perspective considered

### Review Process

**Draft Review:**
1. Self-review using style guide
2. Technical accuracy review by subject matter expert
3. Editorial review for language and clarity
4. User testing with target audience (when possible)

**Maintenance:**
- Review quarterly for accuracy
- Update examples with version changes
- Monitor user feedback and questions
- Track analytics for popular content

## Tools and Resources

### Writing Tools

**Recommended:**
- **Grammarly** or similar for grammar checking
- **Hemingway Editor** for readability
- **Vale** for style linting (configured for this guide)
- **markdownlint** for markdown formatting

### Reference Materials

**Internal:**
- [Contributing Guidelines](guidelines.md)
- [Documentation Templates](../templates/README.md)
- [API Documentation Standards](../api/index.md)

**External:**
- [Plain Language Guidelines](https://www.plainlanguage.gov/guidelines/)
- [Google Developer Documentation Style Guide](https://developers.google.com/style)
- [Markdown Guide](https://www.markdownguide.org/)

## Examples

### Good Documentation Example

```markdown
# Setting Up Redis Cache

Redis provides fast, in-memory caching for the Tracktion analysis service. This guide shows you how to configure Redis for development and production environments.

## Prerequisites

- Docker installed and running
- Analysis service configured (see [Setup Guide](../setup/installation.md))

## Development Setup

1. **Start Redis container:**
   ```bash
   docker run -d --name redis -p 6379:6379 redis:7-alpine
   ```

2. **Configure the analysis service:**
   ```bash
   # Add to .env file
   REDIS_URL=redis://localhost:6379
   ```

3. **Verify connection:**
   ```bash
   redis-cli ping
   # Expected output: PONG
   ```

## Troubleshooting

**Connection refused error:**
If you see "Connection refused", ensure Redis is running:
```bash
docker ps | grep redis
```

If not running, restart the container:
```bash
docker start redis
```

## Next Steps

- [Configure caching policies](caching-configuration.md)
- [Monitor Redis performance](monitoring.md)
```

### Poor Documentation Example

```markdown
# Redis Setup

Redis is fast. You should use it.

## Setup

Install redis and start it. Then configure your app.

```bash
redis-cli
```

It should work. If not, restart it.

## Other stuff

See other docs for more info.
```

## Conclusion

Following this style guide ensures that Tracktion documentation is:
- **Consistent** across all services and components
- **Accessible** to users with different technical backgrounds
- **Maintainable** as the project evolves
- **Professional** in appearance and tone

For questions about this style guide or specific writing situations, create an issue in the documentation repository or ask in the development team channel.

---

**Last Updated:** 2024-01-09
**Next Review:** 2024-04-09
**Maintained By:** Documentation Team
