# Contributing to Documentation

This guide explains how to contribute to Tracktion's documentation, whether you're fixing a typo, adding a new guide, or improving existing content.

## Quick Start for Contributors

### 1. Find Something to Work On

**Good First Contributions:**
- Fix typos or grammatical errors
- Update outdated information
- Add missing code examples
- Improve clarity of existing instructions
- Add troubleshooting tips from your experience

**Documentation Needs:**
Check our [documentation issues](https://github.com/your-org/tracktion/labels/documentation) or look for:
- 🔍 Missing documentation for new features
- 📝 Outdated setup instructions
- 🐛 Broken links or examples
- 🎯 User feedback requesting clarification

### 2. Set Up Your Environment

**Prerequisites:**
- Git and GitHub account
- Python 3.11+ with uv installed
- Basic familiarity with Markdown

**Setup Steps:**
```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/tracktion.git
cd tracktion

# Install dependencies
uv sync

# Install documentation dependencies
uv pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-git-committers-plugin-2 mkdocs-git-authors-plugin mkdocs-macros-plugin mkdocs-include-markdown-plugin mkdocs-jupyter mike

# Start local documentation server
uv run mkdocs serve

# Open http://localhost:8000 in your browser
```

### 3. Make Your Changes

**Documentation is located in:**
- `docs/` - Main documentation content
- `services/*/README.md` - Service-specific documentation
- `README.md` - Project overview

**Live Preview:**
The local server automatically reloads when you make changes, so you can see your updates immediately.

## Types of Contributions

### Content Updates

**Fixing Existing Content:**
1. Identify the file to update
2. Make your changes following the [Style Guide](style-guide.md)
3. Test any code examples
4. Check for broken links
5. Submit a pull request

**Adding New Content:**
1. Choose the appropriate [template](../templates/README.md)
2. Create the new file in the correct location
3. Add the page to `mkdocs.yml` navigation
4. Follow content guidelines
5. Submit a pull request

### Code Examples and Tutorials

**Requirements for Code Examples:**
- Must be functional and tested
- Include all necessary imports and setup
- Use realistic data and variable names
- Work with current software versions
- Include expected output where helpful

**Tutorial Guidelines:**
- Start with clear learning objectives
- Include prerequisites and time estimates
- Break complex tasks into numbered steps
- Provide troubleshooting for common issues
- End with next steps or related resources

### API Documentation

**Updating API Docs:**
1. Auto-generated docs are in `docs/api/`
2. Update source code docstrings for auto-generated content
3. Run `uv run python scripts/generate_docs.py` to regenerate
4. For manual API docs, follow the [API template](../templates/api-documentation.md)

## Writing Guidelines

### Style and Tone

**Writing Style:**
- **Clear and Concise**: Use simple, direct language
- **User-Focused**: Write from the user's perspective
- **Action-Oriented**: Tell users what to do, not just what something is
- **Consistent**: Use the same terms and formatting throughout

**Tone:**
- Professional but approachable
- Helpful and encouraging
- Respectful of different skill levels
- Avoid jargon without explanation

### Content Structure

**Document Organization:**
```markdown
# Page Title (H1)

Brief overview paragraph explaining what this page covers.

## What You'll Learn (if tutorial)

- Learning objective 1
- Learning objective 2

## Prerequisites (if applicable)

- Required knowledge
- Required software
- Links to setup guides

## Main Content

### Section 1 (H2)
Content organized logically...

#### Subsection (H3)
Detailed information...

## Troubleshooting (if applicable)

Common issues and solutions...

## Next Steps

- Link to related content
- Suggested follow-up actions

## Related Resources

- Links to related documentation
- External resources
```

### Formatting Standards

**Markdown Best Practices:**

```markdown
# Use descriptive headings
## Configure Database Connection

# Include working code examples
```bash
# Start the development server
uv run uvicorn main:app --reload --port 8001
```

# Use tables for structured information
| Parameter | Required | Description |
|-----------|----------|-------------|
| `api_key` | Yes | Your API authentication key |

# Use admonitions for important info
!!! warning "Important"
    Always backup your database before running migrations.

# Include realistic examples
```python
from tracktion.analysis import BPMDetector

detector = BPMDetector(confidence_threshold=0.8)
result = detector.analyze("/path/to/song.mp3")
```
```

## Review Process

### Self-Review Checklist

Before submitting your contribution:

**Content Quality:**
- [ ] Information is accurate and current
- [ ] Code examples have been tested
- [ ] Language is clear and concise
- [ ] Follows the [Style Guide](style-guide.md)
- [ ] Uses appropriate formatting
- [ ] Includes necessary context

**Technical Accuracy:**
- [ ] All commands work as documented
- [ ] Links are functional
- [ ] Version-specific information is current
- [ ] Examples use realistic data
- [ ] Dependencies are correctly specified

**User Experience:**
- [ ] Instructions are easy to follow
- [ ] Prerequisites are clearly stated
- [ ] Troubleshooting covers common issues
- [ ] Next steps are provided
- [ ] Appropriate for target audience

### Pull Request Guidelines

**PR Title Format:**
```
docs: Add troubleshooting guide for Redis connection issues
docs: Update API authentication examples
docs: Fix typo in installation instructions
```

**PR Description Template:**
```markdown
## Description
Brief description of what this PR changes or adds.

## Type of Change
- [ ] Fix (typo, broken link, incorrect information)
- [ ] Update (refresh existing content, update examples)
- [ ] New content (new guide, tutorial, reference)
- [ ] Restructure (reorganize existing content)

## Testing
- [ ] All code examples tested and working
- [ ] Links verified and functional
- [ ] Local documentation site builds without errors
- [ ] Content reviewed for accuracy

## Documentation Impact
- [ ] No navigation changes needed
- [ ] Added new page to navigation
- [ ] Updated related pages
- [ ] Added cross-references where appropriate

## Checklist
- [ ] Followed [Style Guide](docs/contributing/style-guide.md)
- [ ] Used appropriate [template](docs/templates/README.md)
- [ ] Self-reviewed using checklist above
- [ ] Ready for review

## Screenshots (if applicable)
Include screenshots of new or significantly changed content.
```

### Review Process

**What Reviewers Look For:**

1. **Technical Accuracy**
   - Code examples work correctly
   - Information is current and accurate
   - Dependencies and versions are correct

2. **User Experience**
   - Content is organized logically
   - Instructions are clear and complete
   - Appropriate level of detail for audience

3. **Style Consistency**
   - Follows established style guide
   - Uses consistent formatting
   - Maintains professional tone

4. **Completeness**
   - All necessary information included
   - Proper context provided
   - Links to related resources

**Review Timeline:**
- Simple fixes (typos, formatting): 1-2 days
- Content updates: 3-5 days
- New content: 5-10 days
- Major restructuring: 1-2 weeks

## Advanced Contributions

### Documentation Infrastructure

**Working with MkDocs:**
```bash
# Test documentation build
uv run mkdocs build --strict

# Deploy documentation locally with versioning
./scripts/deploy_docs.sh dev --local

# Generate API documentation
uv run python scripts/generate_docs.py
```

**Adding Navigation:**
Edit `mkdocs.yml` to add new pages:
```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Your New Page: getting-started/your-new-page.md
```

### Automation and Tooling

**Auto-Generation:**
- API docs are generated from code docstrings
- Service READMEs can use templates
- Configuration docs can be extracted from code

**CI/CD Integration:**
- Documentation builds on every PR
- Broken links are detected automatically
- Spelling and grammar are checked
- Style guide compliance is enforced

### Internationalization (Future)

**Preparing for Multiple Languages:**
- Keep text in separate files when possible
- Avoid embedding text in images
- Use clear, simple language
- Document cultural considerations

## Common Tasks

### Adding a New Service Guide

1. **Create the guide:**
   ```bash
   cp docs/templates/service-readme.md services/YOUR_SERVICE/README.md
   ```

2. **Fill in the template:**
   - Replace all `{{ placeholder }}` variables
   - Add service-specific information
   - Include working examples

3. **Add to navigation:**
   Update `mkdocs.yml` to include your service in the Services section

4. **Test locally:**
   ```bash
   uv run mkdocs serve
   ```

### Updating API Documentation

1. **Update code docstrings:**
   Add or improve docstrings in the service code

2. **Regenerate docs:**
   ```bash
   uv run python scripts/generate_docs.py
   ```

3. **Review generated content:**
   Check `docs/api/` for the updated content

4. **Add manual documentation:**
   Create detailed guides in `docs/api/` as needed

### Creating a Tutorial

1. **Choose the right template:**
   Use `docs/templates/tutorial-template.md`

2. **Plan your content:**
   - Define learning objectives
   - Identify prerequisites
   - Break into logical steps
   - Plan examples and exercises

3. **Test everything:**
   Follow your own tutorial from start to finish

4. **Get feedback:**
   Have someone else try your tutorial

### Fixing Broken Links

1. **Find broken links:**
   The CI system will report broken links, or use:
   ```bash
   uv run mkdocs build --strict
   ```

2. **Update links:**
   - Fix internal links to use correct paths
   - Update external links that have moved
   - Remove links to deleted content

3. **Test the fix:**
   Verify links work in the local server

## Getting Help

### Resources

**Documentation:**
- [Style Guide](style-guide.md) - Writing standards
- [Templates](../templates/README.md) - Document templates
- [MkDocs Documentation](https://www.mkdocs.org/) - Tool documentation

**Community:**
- GitHub Discussions for questions
- Documentation team in Slack
- Office hours (see team calendar)

### Common Questions

**Q: How do I know what documentation is needed?**
A: Check GitHub issues labeled "documentation", ask in team channels, or look for gaps in your own experience using the project.

**Q: Can I contribute if I'm not a developer?**
A: Absolutely! Documentation benefits greatly from the perspective of users who are learning the system.

**Q: How do I test API examples without access to production?**
A: Use the development environment setup instructions, or ask the team for access to a test environment.

**Q: What if I'm not sure about technical accuracy?**
A: Do your best and note any uncertainties in your PR. The review process will catch technical issues.

## Recognition

Contributors to documentation are recognized in:
- Git commit history and GitHub contributor graphs
- Quarterly contributor highlights
- Annual project reports
- Documentation page attribution (for major contributions)

Thank you for helping make Tracktion's documentation better! 📚

---

**Questions about contributing?**
- Create a [GitHub issue](https://github.com/your-org/tracktion/issues/new?labels=documentation,question)
- Ask in the [#documentation Slack channel](https://tracktion.slack.com)
- Check our [FAQ section](../reference/faq.md)
