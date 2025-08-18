# Compatibility Matrix

This document tracks the compatibility of `llama-stack-provider-lmeval` with different versions of [llama-stack](https://github.com/llamastack/llama-stack) and its dependencies across releases.

## Version Compatibility Table

| Provider Version | Llama-Stack Version | Python Version | Key Dependencies | Status | Notes |
|------------------|---------------------|----------------|------------------|---------|-------|
| 0.2.0 | >=0.2.5 | >=3.12 | kubernetes, fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, uvicorn, ipykernel | Current | Latest release with enhanced compatibility |
| 0.1.8 | >=0.2.5 | >=3.12 | kubernetes, fastapi, opentelemetry-api, opentelemetry-exporter-otlp, aiosqlite, uvicorn, ipykernel |  | Initial stable release |

## Dependency Details

### Core Dependencies

#### Version 0.2.0
- **llama-stack**: >=0.2.5
- **kubernetes**: Latest compatible
- **fastapi**: Latest compatible
- **opentelemetry-api**: Latest compatible
- **opentelemetry-exporter-otlp**: Latest compatible
- **aiosqlite**: Latest compatible
- **uvicorn**: Latest compatible
- **ipykernel**: Latest compatible

#### Version 0.1.8
- **llama-stack**: >=0.2.5
- **kubernetes**: Latest compatible
- **fastapi**: Latest compatible
- **opentelemetry-api**: Latest compatible
- **opentelemetry-exporter-otlp**: Latest compatible
- **aiosqlite**: Latest compatible
- **uvicorn**: Latest compatible
- **ipykernel**: Latest compatible

### Development Dependencies

Both versions include the same development dependencies:
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Linting and formatting
- **pre-commit**: Git hooks
- **mypy**: Type checking
- **types-PyYAML**: Type stubs for PyYAML
- **types-requests**: Type stubs for requests

## Container Compatibility

The provider is tested and compatible with:
- **Base Image**: `registry.access.redhat.com/ubi9/python-312:latest`
- **Llama-Stack Version**: 0.2.16 (in container builds)
- **Additional Runtime Dependencies**: torch, sentence-transformers, sqlalchemy, and others as specified in the Containerfile

## Breaking Changes

### Version 0.2.0
- No breaking changes from 0.1.8
- Enhanced compatibility with newer llama-stack versions

### Version 0.1.8
- Initial release
- No breaking changes

## Future Planning

This compatibility matrix will be updated with each new release to include:
- New llama-stack version compatibility
- Dependency updates and changes
- Breaking changes and migration notes
- Container compatibility updates
- Testing status and known issues
