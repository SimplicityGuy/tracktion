---
name: Documentation Change
about: Template for documentation-related pull requests
title: 'docs: '
labels: documentation
assignees: ''
---

## Description
<!-- Provide a clear and concise description of what this PR changes or adds -->

## Type of Documentation Change
<!-- Check all that apply -->
- [ ] **Fix** - Typo, broken link, incorrect information
- [ ] **Update** - Refresh existing content, update examples, version changes
- [ ] **New Content** - New guide, tutorial, reference documentation
- [ ] **Restructure** - Reorganize existing content, improve navigation
- [ ] **Auto-generated** - Updates from code docstrings or automated tools

## Motivation and Context
<!-- Why is this change needed? What problem does it solve? -->
<!-- Link to any related issues using "Fixes #issue-number" or "Relates to #issue-number" -->

## Changes Made
<!-- Describe the specific changes made -->
-
-
-

## Testing and Validation
<!-- Confirm you've tested your changes -->
- [ ] All code examples have been tested and work correctly
- [ ] Links have been verified and are functional
- [ ] Local documentation site builds without errors (`uv run mkdocs build --strict`)
- [ ] Content has been spell-checked and grammar-checked
- [ ] Screenshots/images are optimized and have alt text

## Documentation Quality Checklist
<!-- Ensure your changes meet our documentation standards -->

### Content Quality
- [ ] Information is accurate and up-to-date
- [ ] Language is clear and concise
- [ ] Appropriate level of detail for target audience
- [ ] Includes necessary context and background
- [ ] Follows the [Style Guide](../docs/contributing/style-guide.md)

### Structure and Navigation
- [ ] Content is organized logically
- [ ] Uses appropriate heading hierarchy (H1 → H2 → H3)
- [ ] Added to navigation in `mkdocs.yml` (if new page)
- [ ] Cross-references to related content added
- [ ] Follows established document templates

### Code and Examples
- [ ] All code examples are functional and tested
- [ ] Examples use realistic data and meaningful variable names
- [ ] Includes necessary imports and setup context
- [ ] Code follows project coding standards
- [ ] Version requirements specified where relevant

### User Experience
- [ ] Prerequisites clearly stated
- [ ] Step-by-step instructions are easy to follow
- [ ] Troubleshooting section included (where appropriate)
- [ ] Next steps or related resources provided
- [ ] Accessible to target audience skill level

## Impact Assessment
<!-- Describe the impact of your changes -->

### Pages Affected
<!-- List all documentation pages that are new, modified, or affected -->
-
-

### Navigation Changes
<!-- Describe any changes to site navigation -->
- [ ] No navigation changes
- [ ] Added new pages to navigation
- [ ] Reorganized existing navigation
- [ ] Updated navigation labels

### Breaking Changes
<!-- Any changes that might break existing links or user workflows -->
- [ ] No breaking changes
- [ ] Changed file paths or URLs
- [ ] Removed or renamed sections
- [ ] Changed command syntax or examples

## Screenshots
<!-- Include screenshots of new or significantly changed content -->
<!-- Use GitHub's drag-and-drop to add images -->

### Before (if updating existing content)
<!-- Screenshot of content before your changes -->

### After
<!-- Screenshot of content after your changes -->

## Review Requests
<!-- Who should review this PR? -->
- [ ] **Technical Review** - Subject matter expert for accuracy
- [ ] **Editorial Review** - Documentation team for style and clarity
- [ ] **User Testing** - Have someone from target audience try instructions

### Specific Review Areas
<!-- Highlight areas where you'd like focused review -->
-
-

## Additional Context
<!-- Any additional information that reviewers should know -->
<!-- Links to related PRs, issues, or external resources -->

## Pre-Submission Checklist
<!-- Final check before submitting -->
- [ ] PR title follows format: `docs: brief description of change`
- [ ] All template sections completed
- [ ] Self-reviewed changes using documentation quality checklist
- [ ] Tested locally with `uv run mkdocs serve`
- [ ] Ready for team review

---

<!--
Thank you for contributing to Tracktion documentation!
Your efforts help make the project more accessible to everyone.

For questions about the documentation process:
- Check the [Contributing Guide](../docs/contributing/documentation.md)
- Ask in #documentation Slack channel
- Create a GitHub issue with the 'documentation' label
-->
