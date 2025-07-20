# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] - 2025-01-27

### Changed
- Updated mojentic dependency to version >=0.7.4

## [0.4.1] - 2025-01-27

### Changed
- Updates to documentation

## [0.4.0] - 2025-07-19

### Added
- Progress tracking system with CLI integration for better user feedback during operations
- Comprehensive command pattern tests and modular base implementations
- Robust conflict resolution strategies with "not a conflict" support and editor-based input
- Architectural backlog and testing specifications for future development

### Changed
- Expanded `content` field maximum length to 2000 characters for larger content handling
- Enhanced conflict resolution with new options and improved progress tracking
- Refactored conflict detection to use `asyncio.gather` for improved batch processing performance

### Fixed
- Enforced chunk size constraints and clarified granularity levels for better content processing
- Cleaned up `ingest` command formatting for improved user experience
- Removed deprecated debugging scripts to reduce codebase complexity

## [0.3.1] - 2025-01-27

### Fixed
- Code quality improvements and maintenance updates
- Verified all 146 tests pass successfully
- Ensured codebase stability for production use

## [0.3.0] - 2025-07-06

### Added
- **Project Context Isolation**: Prevent cross-project knowledge contamination with project-aware parameters
  - `--project-id` and `--project-name` parameters for ingest command
  - `--project-ids` and `--exclude-projects` parameters for assemble command
  - Project-aware filtering in slice command
- **CRAFT-Aware CLI Enhancements**: Enhanced all major commands with CRAFT framework parameters
  - Slice command now supports `--granularity`, `--domains`, `--project-ids`, `--exclude-projects`, and `--authority-level`
  - Assemble command enhanced with project filtering and token optimization
  - Quarantine command suite for conflict management
- **ChunkingEngine**: Semantic boundary detection and intelligent content processing (988 lines of code)
- **KnowledgeQuarantine System**: Complete isolation mechanism for conflicting knowledge (11,810 bytes)
- **Enhanced Documentation**: Updated all usage documentation with new CRAFT-aware capabilities
  - Updated README.md with new features and correct CLI examples
  - Updated docs/index.md with accurate command examples
  - Completely revised docs/assembling-copilot-instructions.md with realistic examples
  - Enhanced docs/ingesting-prompts.md with project-aware capabilities

### Changed
- Slice command now extracts content categories (why, who, what, how) instead of tag-based slicing
- All CLI commands updated to use correct parameter syntax
- Documentation examples updated to reflect actual CLI implementation

### Enhanced
- Token optimization with `--token-budget` and `--quality-threshold` parameters
- Domain and granularity filtering across all relevant commands
- Authority-level filtering for quality control

## [0.2.0] - 2025-07-05

### Added
- Initial release preparation
- Core functionality for context mixing across GenAI coding assistants
- Command-line interface with Typer
- Knowledge store with ChromaDB integration
- Prompt ingestion and assembly capabilities
- Quarantine system for conflict resolution
- Git integration for project context
- Comprehensive test suite with 126 tests
- Documentation with MkDocs

### Changed
- Version bump from 0.1.0 to 0.2.0 for release preparation

## [0.1.0] - Initial Development
- Initial project setup and development
