# Integration Tests

End-to-end integration tests for the InfiniteHash protocol.

## Running Tests

```bash
# Run all integration tests
pytest app/src/tests/integration/ -v

# Run specific test
pytest app/src/tests/integration/test_multiprocess_scenario.py -v

# Run with markers
pytest -m integration -v
pytest -m "integration and not slow" -v

# With verbose output and logs
pytest app/src/tests/integration/ -v -s
```

## Test Structure

- **conftest.py** - Shared fixtures and configuration for integration tests
- **test_multiprocess_scenario.py** - Full multi-window auction cycle test

## Environment Variables

- `VERBOSE_LOGS=1` - Enable debug logging from websockets/httpx/httpcore

## Markers

- `@pytest.mark.integration` - Marks end-to-end integration tests
- `@pytest.mark.slow` - Marks slow-running tests
- `@pytest.mark.asyncio` - Marks async tests (requires pytest-asyncio)
- `@pytest.mark.django_db(transaction=False)` - Required for multiprocess tests (allows committed data to be visible across processes)

## Architecture

Integration tests use a multiprocess architecture:
- **Main process**: Runs pytest and orchestrates the scenario
- **Simulator process**: Runs the blockchain simulator (HTTP + RPC servers)
- **Worker processes**: Run validator and miner workers with isolated databases

All processes share the same test database but use PostgreSQL schemas for isolation.
