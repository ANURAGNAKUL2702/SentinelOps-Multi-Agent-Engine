# 🎮 Autonomous War-Room Simulator

AI-powered incident response simulation with end-to-end root cause analysis, automated reporting, and historical analytics.

## Architecture Overview

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌───────────┐   ┌──────┐
│  Simulation  │──▶│ Observability│──▶│  7-Agent Pipeline │──▶│ Reporting │──▶│  DB  │
│  Engine      │   │  (Metrics,   │   │  (Orchestrator)   │   │ (HTML,    │   │      │
│  (Phases 1-3)│   │   Logs, Deps)│   │  (Phases 5-11)    │   │  JSON, …) │   │      │
└──────────────┘   └──────────────┘   └──────────────────┘   └───────────┘   └──────┘
       Phase 1-3         Phase 4             Phase 5-11          Phase 12       Phase 12
```

**Phase 13 (this layer)** wires all 12 phases into a single CLI application.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simulation
python main.py run --scenario database_timeout --format html

# List available scenarios
python main.py list-scenarios

# See all commands
python main.py --help
```

## CLI Commands

| Command            | Description                                      |
|--------------------|--------------------------------------------------|
| `run`              | Run an incident simulation and generate reports  |
| `list-scenarios`   | List all available incident scenarios            |
| `analyze`          | Analyze historical incidents from the database   |
| `dashboard`        | Generate an executive dashboard HTML             |
| `serve`            | Start the REST API server                        |
| `validate`         | Validate the configuration file                  |
| `interactive`      | Interactive mode with guided prompts             |
| `export-metrics`   | Export pipeline metrics (Prometheus format)       |
| `version`          | Show version and dependency information          |

### Examples

```bash
# Run with multiple report formats
python main.py run -s memory_leak -f html -f json

# Analyze last 7 days, filter by severity
python main.py analyze --days 7 --severity P1

# Generate dashboard
python main.py dashboard --days 30 -o dashboard.html

# Start API server
python main.py serve --port 8000

# Validate configuration
python main.py validate --config config.yaml

# Interactive mode
python main.py interactive
```

## Available Scenarios

| Scenario           | Description                                  | Severity    |
|--------------------|----------------------------------------------|-------------|
| `memory_leak`      | Gradual heap exhaustion on payment-service   | P2 MEDIUM   |
| `cpu_spike`        | Sudden CPU saturation on fraud scoring       | P1 HIGH     |
| `database_timeout` | Primary database becomes unresponsive        | P0 CRITICAL |
| `network_latency`  | Upstream network degradation at API gateway  | P1 HIGH     |

## Configuration

All settings live in `config.yaml`. Key sections:

```yaml
system:
  log_level: INFO          # DEBUG | INFO | WARNING | ERROR

simulation:
  default_scenario: database_timeout
  duration_minutes: 30

orchestrator:
  pipeline_timeout_seconds: 60
  enable_parallel_execution: true

reporting:
  default_formats: [html, json]
  output_directory: reports
  database:
    url: sqlite:///incidents.db
```

### Environment Variable Overrides

| Variable               | Config Path                    |
|------------------------|--------------------------------|
| `WARROOM_LOG_LEVEL`    | `system.log_level`             |
| `WARROOM_DATABASE_URL` | `reporting.database.url`       |
| `WARROOM_API_ENABLED`  | `api.enabled`                  |
| `WARROOM_API_PORT`     | `api.port`                     |
| `WARROOM_OUTPUT_DIR`   | `reporting.output_directory`   |

## Project Structure

```
app/
├── main.py                    # CLI entry point (Click)
├── config.yaml                # Central configuration
├── conftest.py                # Pytest path setup
├── requirements.txt           # Python dependencies
│
├── integration/               # Phase 13 — integration layer
│   ├── __init__.py
│   ├── config_manager.py      # Pydantic v2 config models + loader
│   ├── logger.py              # Structlog setup with correlation IDs
│   ├── cli.py                 # Rich output helpers & validation
│   └── pipeline.py            # WarRoomPipeline end-to-end orchestrator
│
├── simulation/                # Phases 1-3 — incident simulation
├── observability/             # Phase 4 — metrics, logs, dependencies
├── agents/                    # Phases 5-10 — AI analysis agents
├── orchestrator/              # Phase 11 — pipeline orchestration
├── reporting/                 # Phase 12 — reports, dashboard, API, DB
├── schemas/                   # Shared Pydantic schemas
├── analysis/                  # Analysis utilities
├── validation/                # Validation framework
│
└── tests/
    └── integration/           # Phase 13 integration tests
        ├── test_end_to_end.py # Full pipeline tests
        ├── test_scenarios.py  # Per-scenario tests
        ├── test_cli.py        # CLI command tests
        └── test_config.py     # Configuration tests
```

## Pipeline Flow

1. **Simulation** — Generates realistic incident data (metrics, logs, services, blast radius)
2. **Observability** — Builds queryable metrics/log stores from simulation output
3. **Analysis** — 7 AI agents run in parallel DAG: log analysis → metrics analysis → dependency mapping → hypothesis generation → root cause analysis → validation → incident response
4. **Reporting** — Generates HTML/JSON/Markdown reports with visualizations
5. **Database** — Persists incidents for historical analytics (MTTR, MTTD, SLO compliance)

## Testing

```bash
# Run all tests (1020 tests)
python -m pytest -v

# Run integration tests only (102 tests)
python -m pytest tests/integration/ -v

# Run with coverage
python -m pytest --cov=integration tests/integration/
```

## Exit Codes

| Code | Meaning         |
|------|-----------------|
| 0    | Success         |
| 1    | User error      |
| 2    | System error    |
| 3    | Partial success |

## License

Internal use only.
