# Agent Instructions for Portfolio Returns Repository

## Overview
This document provides guidelines for AI agents working on the Portfolio Returns repository. Follow these instructions to ensure code quality, proper testing, and smooth integration with the existing codebase.

## Core Principles

### 1. Test Before Commit
**CRITICAL**: All code changes MUST be tested before committing to the repository.

- Run all existing tests to ensure no regressions
- Create new tests for new functionality
- Verify the application runs without errors
- Test edge cases and error handling

### 2. Browser Verification Workflow
After each commit/test cycle:

1. **Run the application** in a local development environment
2. **Open Chrome browser** automatically to display the changes
3. **Verify visual and functional changes** work as expected
4. **Document what was changed** in the commit message

### 3. Development Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Understand Goal & Requirements                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Plan Implementation                                  │
│    - Review existing code structure                     │
│    - Identify files to modify/create                    │
│    - Consider dependencies and impacts                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Implement Changes                                    │
│    - Follow existing code style and patterns            │
│    - Add comments for complex logic                     │
│    - Update documentation as needed                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Test Thoroughly                                      │
│    - Run unit tests                                     │
│    - Run integration tests                              │
│    - Manual testing in browser                          │
│    - Verify data integrity                              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Open in Chrome Browser                               │
│    - Launch application                                 │
│    - Demonstrate changes visually                       │
│    - Verify all functionality works                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Commit to Repository                                 │
│    - Write clear, descriptive commit message            │
│    - Reference issue numbers if applicable              │
│    - Push to appropriate branch                         │
└─────────────────────────────────────────────────────────┘
```

## Repository-Specific Guidelines

### Project Structure
```
Portfolio-Returns/
├── app/
│   └── returns_tracker.py    # Main application file
├── trade_reports/            # Trade report data
├── portfolio_history.csv     # Historical portfolio data
├── ticker_mappings.csv       # Ticker symbol mappings
├── requirements.txt          # Python dependencies
└── agents.md                 # This file
```

### Technology Stack
- **Language**: Python
- **Framework**: (Identify from returns_tracker.py - likely Streamlit, Flask, or Dash)
- **Data**: CSV files for portfolio and trade data

### Code Quality Standards

#### Python Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 88 characters (Black formatter standard)

#### Error Handling
- Always use try-except blocks for file operations
- Validate user inputs before processing
- Provide clear error messages to users
- Log errors for debugging purposes

#### Data Integrity
- Validate CSV data before processing
- Handle missing or malformed data gracefully
- Maintain backup copies before modifying data files
- Use pandas for robust data manipulation

### Testing Requirements

#### Before Every Commit
1. **Unit Tests**: Test individual functions
   ```bash
   python -m pytest tests/
   ```

2. **Integration Tests**: Test component interactions
   ```bash
   python -m pytest tests/integration/
   ```

3. **Application Launch**: Verify the app starts without errors
   ```bash
   python app/returns_tracker.py
   # or
   streamlit run app/returns_tracker.py
   ```

4. **Browser Testing**: Open in Chrome and verify:
   - All pages load correctly
   - Data displays accurately
   - Interactive elements function properly
   - No console errors in browser DevTools

### Git Best Practices

#### Commit Messages
Follow the conventional commits format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example**:
```
feat(returns): Add year-over-year comparison chart

- Implemented new chart component using plotly
- Added data aggregation function for yearly returns
- Updated UI to include new chart section
- Tested with historical data from 2020-2025

Closes #42
```

#### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `fix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes

#### Before Pushing
- [ ] All tests pass
- [ ] Code is formatted and linted
- [ ] Documentation is updated
- [ ] Browser verification completed
- [ ] Commit message is clear and descriptive
- [ ] No sensitive data in commits (API keys, passwords, etc.)

### Browser Automation

#### Opening Chrome After Testing
Use one of these methods to automatically open the application in Chrome:

**Method 1: Python webbrowser module**
```python
import webbrowser
webbrowser.get('chrome').open('http://localhost:8501')
```

**Method 2: PowerShell command**
```powershell
Start-Process "chrome.exe" "http://localhost:8501"
```

**Method 3: Streamlit auto-open**
```bash
streamlit run app/returns_tracker.py --browser.gatherUsageStats false
```

### Performance Considerations
- Optimize data loading for large CSV files
- Use caching for expensive computations
- Minimize unnecessary re-renders
- Profile code to identify bottlenecks

### Security Guidelines
- Never commit sensitive data (API keys, credentials)
- Use environment variables for configuration
- Validate and sanitize all user inputs
- Keep dependencies up to date for security patches

### Documentation Requirements
- Update README.md for user-facing changes
- Add inline comments for complex algorithms
- Document API endpoints if applicable
- Maintain changelog for version tracking

## Agent-Specific Instructions

### When Adding New Features
1. Review existing codebase for similar patterns
2. Ensure consistency with current architecture
3. Add appropriate error handling
4. Create tests for the new feature
5. Update documentation
6. Test in browser before committing

### When Fixing Bugs
1. Reproduce the bug consistently
2. Write a test that fails due to the bug
3. Implement the fix
4. Verify the test now passes
5. Check for similar bugs elsewhere
6. Document the fix in commit message

### When Refactoring
1. Ensure all tests pass before starting
2. Make incremental changes
3. Run tests after each change
4. Keep commits small and focused
5. Verify functionality remains unchanged

### When Working with Data
1. Always validate CSV structure
2. Handle missing data gracefully
3. Preserve data integrity
4. Test with edge cases (empty files, malformed data)
5. Backup data before modifications

## Checklist for Each Task

- [ ] Understand the specific goal and requirements
- [ ] Review relevant existing code
- [ ] Plan the implementation approach
- [ ] Implement changes following code standards
- [ ] Write/update tests
- [ ] Run all tests locally
- [ ] Launch application locally
- [ ] Open in Chrome browser
- [ ] Verify changes work as expected
- [ ] Check for console errors
- [ ] Update documentation
- [ ] Write clear commit message
- [ ] Commit and push changes

## Questions to Ask Before Starting
1. What is the specific goal or problem to solve?
2. Are there any constraints or requirements?
3. Which files need to be modified?
4. What tests need to be created or updated?
5. Are there any dependencies to consider?
6. How should success be measured?

## Resources
- Python Documentation: https://docs.python.org/3/
- PEP 8 Style Guide: https://pep8.org/
- Git Best Practices: https://git-scm.com/book/en/v2
- Pandas Documentation: https://pandas.pydata.org/docs/

---

**Remember**: Quality over speed. Take time to test thoroughly and ensure changes work correctly before committing to the repository.
