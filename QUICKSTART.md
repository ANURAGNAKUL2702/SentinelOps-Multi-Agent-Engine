# Autonomous War-Room Simulator — Step-by-Step Quickstart

A beginner-friendly walkthrough to get the project running from scratch.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Navigate](#2-clone--navigate)
3. [Set Up Python Environment](#3-set-up-python-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Verify Installation](#5-verify-installation)
6. [Run Your First Simulation](#6-run-your-first-simulation)
7. [Understand the Output](#7-understand-the-output)
8. [Open the HTML Report](#8-open-the-html-report)
9. [Try All 4 Scenarios](#9-try-all-4-scenarios)
10. [Explore Other Commands](#10-explore-other-commands)
11. [Run the Tests](#11-run-the-tests)
12. [Project Structure Explained](#12-project-structure-explained)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

| Requirement    | Version   | Check with              |
|----------------|-----------|-------------------------|
| **Python**     | 3.11+     | `python --version`      |
| **pip**        | 23+       | `pip --version`         |
| **Git**        | any       | `git --version`         |

> **Windows users**: Make sure Python is added to your PATH during installation.
> **macOS/Linux**: Use `python3` and `pip3` if `python` points to Python 2.

---

## 2. Clone & Navigate

```bash
git clone <your-repo-url> autonomous-warroom-simulator
cd autonomous-warroom-simulator/app
```

All commands below assume you are inside the `app/` directory.

---

## 3. Set Up Python Environment

Create a virtual environment to keep dependencies isolated:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs ~15 packages: Click, Rich, Pydantic, FastAPI, SQLAlchemy, matplotlib, etc.

**Expected output** (last line):
```
Successfully installed click-8.1.x rich-13.x.x pydantic-2.x.x ...
```

---

## 5. Verify Installation

Run these quick checks to confirm everything is ready:

```bash
# Check the CLI loads correctly
python main.py --help
```

**Expected output:**
```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  🎮 Autonomous War-Room Simulator
  ...

Commands:
  analyze         Analyze historical incidents from the database.
  dashboard       Generate an executive dashboard HTML.
  interactive     Interactive mode ...
  list-scenarios  List all available incident scenarios.
  run             Run an incident simulation and generate reports.
  serve           Start the REST API server.
  validate        Validate the configuration file.
  version         Show version and dependency information.
```

```bash
# Validate the config file
python main.py validate
```

```bash
# Show version and installed packages
python main.py version
```

---

## 6. Run Your First Simulation

This is the core command. It runs the **entire pipeline end-to-end**:

```bash
python main.py run --scenario database_timeout --format html
```

**What happens behind the scenes:**

```
Step 1: Simulation Engine         → Generates 240 metrics + 480 logs for 8 microservices
Step 2: Observability Layer       → Builds queryable metrics/log stores
Step 3: 7 AI Agents (in DAG)      → Analyzes the incident:
        ├── Log Agent             → Finds error patterns across services
        ├── Metrics Agent         → Detects CPU/memory/latency anomalies
        ├── Dependency Agent      → Maps blast radius + cascading failures
        ├── Hypothesis Agent      → Generates ranked hypotheses
        ├── Root Cause Agent      → Determines the most likely root cause
        ├── Validation Agent      → Validates the verdict against ground truth
        └── Incident Commander    → Produces runbook + action items
Step 4: Report Generator          → Creates a beautiful HTML report
Step 5: Database                  → Saves the incident for historical tracking
```

**Expected output:**
```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ✅ Incident Analysis Complete                       │
│                                                      │
│  Scenario   : database_timeout                       │
│  Root Cause : Network partition or connectivity      │
│               loss causing service failures  (53%)   │
│  Reports    : reports\<correlation-id>.html          │
│  Duration   : 1.5s                                   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 7. Understand the Output

The pipeline produces:

| Output            | Location                          | Description                           |
|-------------------|-----------------------------------|---------------------------------------|
| **HTML Report**   | `reports/<correlation-id>.html`   | Full incident report with charts      |
| **Database**      | `incidents.db` (SQLite)           | Historical incident records           |
| **Console**       | Terminal                          | Summary panel with root cause         |

**Key fields in the result:**
- **Scenario** — The failure type that was simulated
- **Root Cause** — What the AI agents determined was the cause
- **Confidence %** — How confident the analysis is (0–100%)
- **Duration** — Total pipeline execution time

---

## 8. Open the HTML Report

After running a simulation, open the generated report:

**Windows:**
```powershell
# Open the most recent report in your default browser
start (Get-ChildItem reports/*.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

**macOS:**
```bash
open reports/*.html
```

**Linux:**
```bash
xdg-open reports/*.html
```

The HTML report contains:
- Incident summary (root cause, severity, confidence)
- Anomaly details from each agent
- Remediation plan with runbook steps
- Action items with priorities and owners
- Incident timeline

---

## 9. Try All 4 Scenarios

The simulator ships with 4 pre-built failure scenarios:

| Scenario           | What It Simulates                             | Severity |
|--------------------|-----------------------------------------------|----------|
| `database_timeout` | Primary database becomes unresponsive         | SEV-1    |
| `memory_leak`      | Gradual heap exhaustion on payment-service    | SEV-2    |
| `cpu_spike`        | Sudden CPU saturation on fraud scoring        | SEV-2    |
| `network_latency`  | Upstream network degradation at API gateway   | SEV-1    |

**Run each one:**
```bash
python main.py run -s database_timeout -f html
python main.py run -s memory_leak -f html
python main.py run -s cpu_spike -f html
python main.py run -s network_latency -f html
```

**List scenarios programmatically:**
```bash
python main.py list-scenarios
```

**Generate multiple report formats at once:**
```bash
python main.py run -s database_timeout -f html -f json -f markdown
```

---

## 10. Explore Other Commands

### Analyze Historical Incidents

After running several simulations, query the database:

```bash
# Show analytics for the last 30 days
python main.py analyze

# Filter by time range
python main.py analyze --days 7

# Filter by severity
python main.py analyze --severity SEV-1
```

### Generate Executive Dashboard

```bash
python main.py dashboard
# Output: reports/dashboard.html
```

### Start the REST API

```bash
python main.py serve --port 8000
# API docs at http://localhost:8000/docs
```

### Interactive Mode

Don't want to type flags? Use the guided wizard:

```bash
python main.py interactive
```

### Export Prometheus Metrics

```bash
python main.py export-metrics
```

### Skip Database Saving

```bash
python main.py run -s cpu_spike -f html --no-save-db
```

---

## 11. Run the Tests

The project has **1020 tests** covering all 13 phases:

```bash
# Run all tests
python -m pytest

# Run with short output
python -m pytest -q

# Run with verbose output
python -m pytest -v

# Run only integration tests (102 tests)
python -m pytest tests/integration/ -v

# Run a specific test file
python -m pytest tests/integration/test_end_to_end.py -v

# Run tests matching a keyword
python -m pytest -k "database" -v

# Run with coverage report
python -m pytest --cov=integration --cov-report=term-missing tests/integration/
```

**Expected result:**
```
1020 passed, 1 skipped, 26 warnings in ~8s
```

---

## 12. Project Structure Explained

```
app/
├── main.py                     # ← Entry point. Start here.
├── config.yaml                 # ← All settings (timeouts, formats, DB path)
├── requirements.txt            # ← Python dependencies
├── conftest.py                 # ← Pytest setup
├── QUICKSTART.md               # ← This file
├── README.md                   # ← Full project documentation
│
├── integration/                # Glue layer connecting everything
│   ├── pipeline.py             #   ← Core: runs sim → obs → agents → reports → DB
│   ├── config_manager.py       #   ← Loads & validates config.yaml
│   ├── cli.py                  #   ← Rich terminal output helpers
│   └── logger.py               #   ← Structured logging with correlation IDs
│
├── simulation/                 # Generates fake incident data
│   ├── services.py             #   ← 8 microservices & dependencies
│   ├── metrics_engine.py       #   ← CPU, memory, latency, error rate
│   ├── log_engine.py           #   ← Realistic log messages
│   ├── failure_injector.py     #   ← Injects the failure scenario
│   └── dependency_graph.py     #   ← Service dependency + blast radius
│
├── observability/              # Builds queryable stores from simulation
│
├── agents/                     # 7 AI analysis agents
│   ├── log_agent/              #   ← Error pattern analysis
│   ├── metrics_agent/          #   ← Anomaly + correlation detection
│   ├── dependency_agent/       #   ← Blast radius + cascading failure
│   ├── hypothesis_agent/       #   ← Hypothesis generation + ranking
│   ├── root_cause_agent/       #   ← Final root cause verdict
│   ├── validation_agent/       #   ← Validates verdict vs ground truth
│   └── incident_commander_agent/ # ← Runbook + action items
│
├── orchestrator/               # Runs agents in a DAG with timeouts
│   ├── orchestrator.py         #   ← Pipeline controller
│   ├── execution_engine.py     #   ← Async stage executor
│   └── dag.py                  #   ← Directed Acyclic Graph
│
├── reporting/                  # Report generation + database
│   ├── report_builder.py       #   ← Builds HTML/JSON/Markdown reports
│   ├── database/               #   ← SQLite persistence
│   └── api/                    #   ← FastAPI REST endpoints
│
└── reports/                    # ← Generated reports appear here
```

---

## 13. Troubleshooting

### "No module named 'integration'" or "ModuleNotFoundError"

Make sure you're running commands from the `app/` directory:
```bash
cd autonomous-warroom-simulator/app
python main.py run -s database_timeout -f html
```

### "pip install fails" or "externally-managed-environment"

Use a virtual environment (Step 3 above), or:
```bash
pip install --user -r requirements.txt
```

### VS Code shows "Import could not be resolved" warnings

These are Pylance warnings, **not real errors**. The code runs fine. Fix by:
1. Open Command Palette (`Ctrl+Shift+P`)
2. Search "Python: Select Interpreter"
3. Choose the `.venv` interpreter at `.venv/Scripts/python.exe`

### PowerShell script execution disabled

If `.venv\Scripts\Activate.ps1` fails:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Port already in use (for `serve` command)

```bash
python main.py serve --port 8080    # Try a different port
```

### Want to reset the database?

```bash
# Delete the SQLite file and it will be recreated on next run
del incidents.db        # Windows
rm incidents.db         # macOS/Linux
```

---

## Quick Reference Card

```bash
# ─── Most Common Commands ───────────────────────────────
python main.py run -s database_timeout -f html    # Run simulation
python main.py list-scenarios                      # Show scenarios
python main.py analyze                             # Query history
python main.py dashboard                           # Executive view
python main.py interactive                         # Guided wizard
python main.py validate                            # Check config
python -m pytest -q                                # Run all tests
```
