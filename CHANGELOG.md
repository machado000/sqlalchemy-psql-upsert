# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0]

### Changed
- **Major Architecture Update**: Replaced chunked multi-threaded approach with temporary table + CTE-based MERGE operations
- **Improved Type Hints**: Enhanced type annotations throughout the codebase for better IDE support and static analysis
- **Enhanced Docstrings**: Comprehensive documentation updates with better parameter descriptions and return type documentation

### Technical Improvements
- **Temporary Table Approach**: Uses PostgreSQL temporary tables for staging data before conflict resolution
- **Advanced CTE Logic**: Sophisticated Common Table Expression (CTE) queries for conflict analysis and deduplication
- **Single Transaction Operations**: Entire upsert operation now happens in a single transaction for better consistency
- **Atomic MERGE Operations**: Uses PostgreSQL 15+ MERGE statement for atomic upsert operations

### Removed
- **Multi-threading Support**: Removed chunk-based parallel processing in favor of more reliable single-transaction approach
- **Chunk Size Configuration**: No longer needed with temporary table approach
- **Worker Count Configuration**: Removed due to single-threaded design

## [1.0.3] - Previous Release

### Added
- **Automatic NaN to NULL conversion**: All pandas NaN values (np.nan, pd.NaType, None) are now automatically converted to PostgreSQL NULL values during upsert operations
- **Improved data integrity**: Better handling of missing/null data in DataFrames

### Features
- Multi-constraint conflict detection: Automatically handles primary key, unique constraints, and composite constraints
- Smart conflict filtering: Removes rows that would conflict with multiple existing records
- Automatic NaN to NULL conversion: Seamlessly converts pandas NaN values to PostgreSQL NULL values
- Multi-threaded processing: Parallel chunk processing for large datasets
- Configurable batch sizes: Optimize memory usage and processing speed
- Schema validation: Automatic table and column validation before operations
- Comprehensive error handling: Detailed logging and error reporting
