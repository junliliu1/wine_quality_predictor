# Contributing 

Thank you for your interest in contributing to our data analysis project! This document provides guidelines and instructions for contributing to this repository.

> **Note**: This contributing guide follows industry best practices and academic standards for reproducible data science projects. See [Sources and References](#sources-and-references) for detailed attribution.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Documentation Guidelines](#documentation-guidelines)
- [Testing Requirements](#testing-requirements)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Additional Resources](#additional-resources)
- [Sources and References](#sources-and-references)

## Code of Conduct

*This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org/), version 2.1.*

### Our Pledge
We are committed to providing a welcoming and inspiring community for all. Contributors are expected to:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the project and community
- Show empathy towards other contributors

### Unacceptable Behavior
The following behaviors are considered unacceptable:
- Harassment, discrimination, or offensive comments
- Personal or political attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

## How to Contribute

### Types of Contributions
We welcome various types of contributions:
- **Bug Reports**: Help us identify and fix issues
- **Feature Suggestions**: Propose new analyses or improvements
- **Code Contributions**: Submit bug fixes or new features
- **Documentation**: Improve README, comments, or analysis explanations
- **Data Validation**: Help verify data quality and integrity
- **Visualization Improvements**: Enhance charts and graphs

### First Time Contributors
If you're new to the project:
1. Read the project README and documentation
2. Review existing issues and pull requests
3. Start with issues labeled `good first issue` or `help wanted`
4. Ask questions in issue discussions if you need clarification

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Environment Setup
1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/junliliu1/wine_quality_predictor.git
   cd wine_quality_predictor
   ```

2. **Create a Virtual Environment**
   ```bash
   # Using venv
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   
   # Using conda
   conda create -n project-env python=3.8
   conda activate project-env
   ```

3. **Install Dependencies**
   ```bash
   conda env create -f environment.yml
   conda activate wine_quality_predictor
   ```

4. **Set Up Pre-commit Hooks** (if applicable)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Contribution Workflow

### 1. Create an Issue
Before starting work:
- Check if a similar issue already exists
- Create a new issue describing your proposed change
- Wait for discussion and approval from maintainers

### 2. Create a Branch
```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/bug-description
```

### 3. Make Your Changes
- Write clear, concise code following our standards
- Add or update tests as needed
- Update documentation if required
- Ensure all tests pass

### 4. Commit Your Changes
```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "type: brief description

Detailed explanation of changes (if needed)
- Point 1
- Point 2

Fixes #issue-number"
```

#### Commit Message Format
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Test additions or modifications
- **chore**: Maintenance tasks

### 5. Push Changes
```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters
- Use type hints where appropriate
- Follow [NumPy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html) for documentation

### Code Organization
```python
# Example structure for analysis scripts
import statements
from standard_library import modules
from third_party import packages
from local_modules import functions

# Constants
CONSTANT_NAME = "value"

# Classes (if any)
class ClassName:
    """Docstring describing the class."""
    pass

# Functions
def function_name(param: type) -> return_type:
    """
    Brief description of function.
    
    Parameters:
    -----------
    param : type
        Description of parameter
    
    Returns:
    --------
    return_type
        Description of return value
    """
    pass

# Main execution
if __name__ == "__main__":
    main()
```

### Data Processing Standards
- Always validate input data before processing
- Handle missing values explicitly
- Document data transformations clearly
- Save intermediate results for reproducibility

## Documentation Guidelines

### Code Documentation
- All functions must have docstrings
- Complex logic should include inline comments
- Use clear, descriptive names that self-document

### Jupyter Notebooks
- Include markdown cells explaining each analysis step
- Clear output before committing (unless output is essential)
- Number sections logically
- Include data source citations

### README Updates
Update the README when you:
- Add new dependencies
- Change project structure
- Add new features or analyses
- Modify installation or usage instructions

## Testing Requirements

### Test Coverage
- Write unit tests for all new functions
- Maintain minimum 80% code coverage
- Test edge cases and error conditions

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_specific.py
```

### Test File Structure
```python
# tests/test_module.py
import pytest
from src.module import function_to_test

def test_function_normal_case():
    """Test normal operation."""
    assert function_to_test(input) == expected_output

def test_function_edge_case():
    """Test edge cases."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

## Submitting Changes

### Pull Request Process
1. **Ensure all tests pass locally**
2. **Update documentation** as needed
3. **Create a pull request** with:
   - Clear title describing the change
   - Reference to related issue(s)
   - Description of changes made
   - Screenshots (if UI changes)
   - Test results

### Pull Request Template
```markdown
## Description
Brief description of changes

## Related Issue
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Review Process

### What to Expect
1. **Initial Review**: Within 2-3 business days
2. **Feedback**: Constructive comments and suggestions
3. **Revisions**: You may be asked to make changes
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer will merge once approved

### Review Criteria
- Code quality and style compliance
- Test coverage and passing status
- Documentation completeness
- Performance impact
- Security considerations

### After Merge
- Delete your feature branch
- Update your local main branch
- Close related issues

## Additional Resources

### Helpful Links
- [Issue Tracker](https://github.com/junliliu1/wine_quality_predictor/issues)
- [Project Wiki](https://github.com/junliliu1/wine_quality_predictor/wiki)

### Communication Channels
- **Issues**: For bug reports and feature requests
- **Discussions**: For general questions and ideas
- **Pull Requests**: For code review and contribution discussion

### Learning Resources
- [Git Documentation](https://git-scm.com/doc)
- [Python Best Practices](https://docs.python-guide.org/)
- [Data Science Workflow](https://www.kdnuggets.com/2016/03/data-science-workflow.html)

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues for similar questions
3. Open a new issue with the `question` label
4. Reach out to team members listed in the README

---

## Sources and References

This contributing guide incorporates best practices from:

### Open Source Guides
- [GitHub's Open Source Guides](https://opensource.guide/how-to-contribute/)
- [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/)
- [GitHub Contributing Guide Template](https://github.com/nayafia/contributing-template)

### Python and Data Science Standards
- [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [NumPy Documentation Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Pandas Contributing Guide](https://pandas.pydata.org/docs/development/contributing.html)
- [Scikit-learn Contributing Guide](https://scikit-learn.org/stable/developers/contributing.html)

### Reproducible Research
- [The Turing Way - Guide for Reproducible Research](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html)
- [Best Practices for Scientific Computing (Wilson et al., 2014)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745)
- [Good Enough Practices in Scientific Computing (Wilson et al., 2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)

### Version Control and Collaboration
- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Git Flow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)

### Testing and Quality Assurance
- [pytest Documentation](https://docs.pytest.org/en/stable/)
- [Python Testing 101](https://realpython.com/python-testing/)
- [Test-Driven Development with Python](https://www.obeythetestinggoat.com/)

### Documentation Standards
- [Write the Docs - Documentation Guide](https://www.writethedocs.org/guide/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html)

### Academic and Course-Specific Resources
- [UBC MDS Contributing Guidelines](https://ubc-mds.github.io/resources_pages/contributing/)
- [Software Carpentry - Version Control with Git](https://swcarpentry.github.io/git-novice/)
- [Data Carpentry - Project Organization](https://datacarpentry.org/python-ecology-lesson/00-before-we-start/index.html)

---

Thank you for contributing to our project! Your efforts help make this analysis better for everyone.
