"""main.py — CLI entry point for the Autonomous War-Room Simulator.

Uses **Click** for command parsing and **Rich** for beautiful output.

Usage examples::

    python main.py run --scenario database_timeout --format html json
    python main.py list-scenarios
    python main.py analyze --days 30
    python main.py dashboard
    python main.py serve --port 8000
    python main.py validate
    python main.py version
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import click

from integration.cli import (
    console,
    create_progress,
    display_analytics_summary,
    display_error,
    display_result_panel,
    display_scenarios_table,
    format_duration,
    validate_format,
    validate_scenario,
)
from integration.config_manager import ConfigManager, SystemConfig
from integration.logger import setup_logging, get_logger


# ── Click group ────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="warroom")
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    envvar="WARROOM_CONFIG",
    help="Path to config.yaml.",
    type=click.Path(),
)
@click.pass_context
def cli(ctx: click.Context, config_path: str) -> None:
    """🎮 Autonomous War-Room Simulator

    AI-powered incident response simulation with end-to-end root cause
    analysis, automated reporting, and historical analytics.

    \b
    Quick start:
      python main.py run -s database_timeout -f html
      python main.py list-scenarios
      python main.py --help
    """
    ctx.ensure_object(dict)
    try:
        cfg = ConfigManager.load(config_path)
    except Exception as exc:
        console.print(f"[red]Failed to load config:[/red] {exc}")
        raise SystemExit(2) from exc

    ctx.obj["config"] = cfg
    setup_logging(cfg.system.log_level)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── run ────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--scenario",
    "-s",
    required=True,
    help="Incident scenario name (e.g. database_timeout).",
)
@click.option(
    "--format",
    "-f",
    "formats",
    multiple=True,
    default=("html",),
    help="Report format(s): html, markdown, json, pdf.  Repeatable.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory (default: from config).",
)
@click.option(
    "--no-save-db",
    is_flag=True,
    default=False,
    help="Skip saving to the incident database.",
)
@click.pass_context
def run(
    ctx: click.Context,
    scenario: str,
    formats: Tuple[str, ...],
    output: Optional[str],
    no_save_db: bool,
) -> None:
    """Run an incident simulation and generate reports.

    \b
    Examples:
      python main.py run -s database_timeout -f html
      python main.py run -s memory_leak -f html -f json
      python main.py run -s cpu_spike -f json --no-save-db
    """
    config: SystemConfig = ctx.obj["config"]
    log = get_logger("cli.run")

    # Validate scenario
    if not validate_scenario(scenario, config.simulation.available_scenarios):
        console.print(f"[red]Unknown scenario:[/red] '{scenario}'")
        console.print(
            f"Available: {', '.join(config.simulation.available_scenarios)}",
        )
        raise SystemExit(1)

    # Validate formats
    for fmt in formats:
        if not validate_format(fmt):
            console.print(
                f"[red]Invalid format:[/red] '{fmt}'.  "
                "Valid: html, markdown, json, pdf",
            )
            raise SystemExit(1)

    # Override output dir if provided
    if output:
        config = config.model_copy(
            update={
                "reporting": config.reporting.model_copy(
                    update={"output_directory": output},
                ),
            },
        )

    from integration.pipeline import WarRoomPipeline

    pipeline = WarRoomPipeline(config)

    stage_labels = {
        "simulation": "🎬 Running simulation …",
        "observability": "📊 Building observability data …",
        "analysis": "🤖 Running AI agents (7-stage pipeline) …",
        "reporting": "📄 Generating reports …",
        "database": "💾 Saving to database …",
    }

    progress = create_progress()

    with progress:
        task = progress.add_task("Starting …", total=5)

        def _on_stage(stage: str) -> None:
            progress.update(
                task, description=stage_labels.get(stage, stage),
            )
            progress.advance(task)

        result = pipeline.run_scenario(
            scenario_name=scenario,
            formats=list(formats),
            save_to_db=not no_save_db,
            on_stage=_on_stage,
        )

        # Advance to 100 %
        remaining = 5 - progress.tasks[task].completed
        if remaining > 0:
            progress.advance(task, advance=remaining)

    # Display result
    if result.status in ("SUCCESS", "PARTIAL_SUCCESS"):
        display_result_panel(
            {
                "scenario": result.scenario,
                "root_cause": result.root_cause,
                "confidence": result.confidence,
                "report_paths": result.report_paths,
                "duration": result.execution_time,
            },
        )
        if result.errors:
            console.print(
                f"[yellow]⚠ Warnings:[/yellow] {'; '.join(result.errors)}",
            )
        raise SystemExit(0 if result.status == "SUCCESS" else 3)
    else:
        display_error(
            Exception("; ".join(result.errors) or "Pipeline failed"),
            context=scenario,
        )
        raise SystemExit(2)


# ── list-scenarios ─────────────────────────────────────────────────


@cli.command("list-scenarios")
@click.pass_context
def list_scenarios_cmd(ctx: click.Context) -> None:
    """List all available incident scenarios.

    \b
    Example:
      python main.py list-scenarios
    """
    config: SystemConfig = ctx.obj["config"]
    display_scenarios_table(config.simulation.available_scenarios)


# ── analyze ────────────────────────────────────────────────────────


@cli.command()
@click.option("--days", "-d", default=30, show_default=True, help="Look back N days.")
@click.option("--root-cause", "-r", default=None, help="Filter by root-cause substring.")
@click.option(
    "--severity",
    "-s",
    type=click.Choice(["P0", "P1", "P2", "P3"], case_sensitive=False),
    default=None,
    help="Filter by severity level.",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    days: int,
    root_cause: Optional[str],
    severity: Optional[str],
) -> None:
    """Analyze historical incidents from the database.

    \b
    Examples:
      python main.py analyze --days 30
      python main.py analyze --days 7 --severity P1
    """
    config: SystemConfig = ctx.obj["config"]

    from integration.pipeline import WarRoomPipeline

    pipeline = WarRoomPipeline(config)

    with console.status("[bold green]Analyzing incidents …"):
        analytics = pipeline.analyze_history(
            days=days, root_cause=root_cause, severity=severity,
        )

    display_analytics_summary(analytics)


# ── dashboard ──────────────────────────────────────────────────────


@cli.command()
@click.option("--days", "-d", default=30, show_default=True, help="Look back N days.")
@click.option("--output", "-o", default=None, help="Output file path.")
@click.pass_context
def dashboard(ctx: click.Context, days: int, output: Optional[str]) -> None:
    """Generate an executive dashboard HTML.

    \b
    Example:
      python main.py dashboard --days 30 -o dashboard.html
    """
    config: SystemConfig = ctx.obj["config"]

    from integration.pipeline import WarRoomPipeline

    pipeline = WarRoomPipeline(config)

    with console.status("[bold green]Generating dashboard …"):
        path = pipeline.generate_dashboard(days=days, output_path=output)

    console.print(f"[green]✅ Dashboard saved to:[/green] {path}")


# ── serve ──────────────────────────────────────────────────────────


@cli.command()
@click.option("--host", "-h", default=None, help="Bind host (default: from config).")
@click.option("--port", "-p", default=None, type=int, help="Bind port (default: from config).")
@click.pass_context
def serve(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """Start the REST API server.

    \b
    Example:
      python main.py serve --port 8000
    """
    config: SystemConfig = ctx.obj["config"]
    bind_host = host or config.api.host
    bind_port = port or config.api.port

    console.print(
        f"[bold green]Starting API server on {bind_host}:{bind_port} …[/bold green]",
    )
    console.print("Swagger UI → http://localhost:{0}/docs".format(bind_port))

    try:
        import uvicorn

        from reporting.api.dependencies import init_dependencies
        from reporting.api.server import app
        from reporting.config import ReportingConfig

        rc = ReportingConfig(
            database_url=config.reporting.database.url,
            output_dir=config.reporting.output_directory,
        )
        init_dependencies(rc)

        uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")
    except ImportError:
        console.print("[red]uvicorn is required: pip install uvicorn[/red]")
        raise SystemExit(2)


# ── validate ───────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    help="Config file to validate.",
    type=click.Path(),
)
def validate(config_path: str) -> None:
    """Validate the configuration file.

    \b
    Example:
      python main.py validate --config config.yaml
    """
    try:
        cfg = ConfigManager.load(config_path)
    except FileNotFoundError:
        console.print(f"[red]Config file not found:[/red] {config_path}")
        raise SystemExit(1)
    except Exception as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise SystemExit(1)

    issues = ConfigManager.validate(cfg)
    if issues:
        console.print("[yellow]⚠ Validation issues:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
        raise SystemExit(1)

    console.print("[green]✅ Configuration is valid.[/green]")
    console.print(f"  Version      : {cfg.system.version}")
    console.print(f"  Log level    : {cfg.system.log_level}")
    console.print(f"  Scenarios    : {len(cfg.simulation.available_scenarios)}")
    console.print(f"  Database     : {cfg.reporting.database.url}")
    console.print(f"  API enabled  : {cfg.api.enabled}")


# ── interactive ────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """Interactive mode — choose scenario and options via prompts.

    \b
    Example:
      python main.py interactive
    """
    config: SystemConfig = ctx.obj["config"]

    console.print("\n[bold]🎮 Autonomous War-Room Simulator — Interactive Mode[/bold]\n")

    # Choose scenario
    scenarios = config.simulation.available_scenarios
    console.print("[bold]Available scenarios:[/bold]")
    for i, s in enumerate(scenarios, 1):
        console.print(f"  {i}. {s}")

    choice = click.prompt(
        "Select scenario (number)",
        type=click.IntRange(1, len(scenarios)),
    )
    scenario = scenarios[choice - 1]

    # Choose format
    fmt = click.prompt(
        "Report format",
        type=click.Choice(["html", "markdown", "json", "pdf"]),
        default="html",
    )

    save_db = click.confirm("Save to database?", default=True)

    console.print(f"\nRunning [cyan]{scenario}[/cyan] → [cyan]{fmt}[/cyan] …\n")

    # Invoke run command programmatically
    ctx.invoke(
        run,
        scenario=scenario,
        formats=(fmt,),
        output=None,
        no_save_db=not save_db,
    )


# ── analyze-kaggle ─────────────────────────────────────────────────


@cli.command("analyze-kaggle")
@click.option(
    "--csv",
    "csv_path",
    default="data/real/incident_event_log.csv",
    show_default=True,
    help="Path to Kaggle incident_event_log.csv.",
    type=click.Path(),
)
@click.option(
    "--limit",
    default=50,
    show_default=True,
    help="Max incidents to analyse (0 = all).",
)
@click.option(
    "--output",
    "-o",
    default="reports/kaggle",
    show_default=True,
    help="Output directory for results.",
)
@click.option(
    "--show-quality",
    is_flag=True,
    default=False,
    help="Print the data-quality scorecard.",
)
@click.option(
    "--filter-priority",
    is_flag=True,
    default=False,
    help="Analyse only P1/P2 (Critical/High) incidents.",
)
@click.pass_context
def analyze_kaggle(
    ctx: click.Context,
    csv_path: str,
    limit: int,
    output: str,
    show_quality: bool,
    filter_priority: bool,
) -> None:
    """Analyse real Kaggle IT incidents with the 7-agent pipeline.

    \b
    Examples:
      python main.py analyze-kaggle --limit 10 --show-quality
      python main.py analyze-kaggle --csv data/real/incident_event_log.csv --limit 0
      python main.py analyze-kaggle --filter-priority --limit 20
    """
    import json
    import time as _time

    config: SystemConfig = ctx.obj["config"]

    try:
        from integrations.real_data.kaggle_parser import KaggleIncidentParser
    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        console.print("Install with: pip install pandas")
        raise SystemExit(2)

    # ── Step 1: Parse ──────────────────────────────────────────────
    parser = KaggleIncidentParser()
    console.print(f"\n[bold]Kaggle IT Incidents Analysis[/bold]\n")
    console.print(f"  Dataset : {csv_path}")

    with console.status("[bold green]Parsing incident event log …"):
        try:
            all_incidents = parser.parse_file(csv_path)
        except FileNotFoundError:
            console.print(f"[red]File not found:[/red] {csv_path}")
            raise SystemExit(1)

    report = parser.generate_data_quality_report()

    console.print(f"  Events  : {report.total_rows:,}")
    console.print(f"  Unique  : {report.unique_incidents:,}")
    console.print(
        f"  Parseable: {len(all_incidents):,}  "
        f"({report.parsing_success_rate:.1%} success)",
    )

    # ── Step 2: Quality report ─────────────────────────────────────
    if show_quality:
        console.print("\n[bold]Data Quality Scorecard[/bold]")
        console.print(f"  Parse time       : {report.parse_time_seconds:.2f}s")
        console.print(f"  Avg events/inc   : {report.avg_events_per_incident:.1f}")
        console.print(f"  Reopened          : {report.reopened_incidents:,}")
        console.print(f"  Analyzable        : {report.analyzable_incidents:,}")
        console.print(f"  Non-technical     : {report.excluded_non_technical:,}")
        console.print("\n  [bold]Field Completeness:[/bold]")
        for fld, pct in sorted(report.field_completeness.items()):
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            console.print(f"    {fld:22s} {bar} {pct:.1%}")
        console.print("\n  [bold]Category Distribution:[/bold]")
        for cat, cnt in sorted(
            report.category_distribution.items(),
            key=lambda x: -x[1],
        ):
            pct_val = cnt / len(all_incidents) * 100
            console.print(f"    {cat:15s}: {cnt:6,}  ({pct_val:.1f}%)")
        console.print("\n  [bold]Priority Distribution:[/bold]")
        for sev, cnt in sorted(
            report.priority_distribution.items(),
            key=lambda x: -x[1],
        ):
            pct_val = cnt / len(all_incidents) * 100
            console.print(f"    {sev:12s}: {cnt:6,}  ({pct_val:.1f}%)")
        if report.warnings:
            console.print("\n  [yellow]Warnings:[/yellow]")
            for w in report.warnings:
                console.print(f"    ⚠ {w}")

    # ── Step 3: Filter ─────────────────────────────────────────────
    technical = [inc for inc in all_incidents if inc.is_technical]
    if filter_priority:
        technical = [inc for inc in technical if inc.priority_num <= 2]
        console.print(
            f"\n  Filtered to P1/P2: {len(technical):,} incidents",
        )

    if limit > 0:
        technical = technical[:limit]

    console.print(f"\n  Analysing: {len(technical)} incidents\n")

    if not technical:
        console.print("[yellow]No incidents to analyse.[/yellow]")
        raise SystemExit(0)

    # ── Step 4: Generate logs & ground truth ───────────────────────
    ground_truth = parser.get_ground_truth(technical)

    # ── Step 5: Run pipeline per incident ──────────────────────────
    from integration.pipeline import WarRoomPipeline
    from integration.logger import new_correlation_id
    from observability import build_observability_from_simulation

    pipeline = WarRoomPipeline(config)
    results_list = []
    correct = 0
    total_confidence = 0.0
    total_time = 0.0

    progress = create_progress()
    with progress:
        task = progress.add_task("Analysing …", total=len(technical))

        for inc in technical:
            inc_logs = parser.convert_to_logs([inc])
            sim_output = parser.to_simulation_output(inc, inc_logs)
            corr_id = new_correlation_id()  # valid UUID for orchestrator

            t0_inc = _time.perf_counter()
            try:
                obs_data = build_observability_from_simulation(sim_output)
                orch_result = pipeline.run_orchestrator(
                    sim_output, obs_data,
                    correlation_id=corr_id,
                )

                # Extract prediction
                rca = getattr(orch_result.agent_outputs, "root_cause_output", None)
                predicted = ""
                conf = 0.0
                if rca is not None:
                    predicted = str(
                        getattr(rca, "root_cause", "")
                        or getattr(rca, "primary_hypothesis", ""),
                    )
                    conf = float(
                        getattr(rca, "confidence", 0.0)
                        or getattr(rca, "overall_confidence", 0.0)
                    )
                if not predicted:
                    predicted = sim_output.get("root_cause", "unknown")
                    conf = 0.5

                elapsed = _time.perf_counter() - t0_inc

                # Match against ground truth (fuzzy — keyword overlap)
                gt = ground_truth[inc.incident_id]
                pred_words = set(predicted.lower().replace("_", " ").split())
                is_correct = False
                for va in gt.valid_answers:
                    va_words = set(va.lower().replace("_", " ").split())
                    # Match if ≥60% of ground-truth keywords appear in prediction
                    if va_words and len(va_words & pred_words) / len(va_words) >= 0.6:
                        is_correct = True
                        break
                    # Or if the whole answer appears (with underscores removed)
                    va_norm = va.lower().replace("_", " ")
                    if va_norm in predicted.lower():
                        is_correct = True
                        break

                if is_correct:
                    correct += 1
                    marker = "[green]CORRECT[/green]"
                else:
                    marker = f"[red]WRONG[/red] (expected: {gt.primary_cause})"

                total_confidence += conf
                total_time += elapsed

                results_list.append({
                    "incident_id": inc.incident_id,
                    "category": inc.domain_category,
                    "predicted": predicted,
                    "confidence": round(conf, 3),
                    "expected": gt.primary_cause,
                    "correct": is_correct,
                    "time_seconds": round(elapsed, 2),
                })

                progress.console.print(
                    f"  {'✅' if is_correct else '❌'} {inc.incident_id}: "
                    f"{predicted} ({conf:.0%}) — {marker}",
                )

            except Exception as exc:
                elapsed = _time.perf_counter() - t0_inc
                total_time += elapsed
                results_list.append({
                    "incident_id": inc.incident_id,
                    "category": inc.domain_category,
                    "predicted": "ERROR",
                    "confidence": 0.0,
                    "expected": ground_truth[inc.incident_id].primary_cause,
                    "correct": False,
                    "time_seconds": round(elapsed, 2),
                    "error": str(exc),
                })
                progress.console.print(
                    f"  ⚠ {inc.incident_id}: [red]ERROR[/red] — {exc}",
                )

            progress.advance(task)

    # ── Step 6: Summary ────────────────────────────────────────────
    n = len(results_list)
    accuracy = correct / n if n > 0 else 0.0
    avg_conf = total_confidence / n if n > 0 else 0.0
    avg_time = total_time / n if n > 0 else 0.0

    console.print("\n" + "═" * 56)
    console.print("[bold]Results[/bold]")
    console.print("═" * 56)
    console.print(f"  Total analysed  : {n}")
    console.print(f"  Correct         : {correct}")
    console.print(f"  Wrong           : {n - correct}")
    console.print(f"  Accuracy        : {accuracy:.1%}")
    console.print(f"  Avg confidence  : {avg_conf:.1%}")
    console.print(f"  Avg time/inc    : {avg_time:.2f}s")

    # ── Step 7: Save results ───────────────────────────────────────
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "dataset": csv_path,
        "total_events": report.total_rows,
        "unique_incidents": report.unique_incidents,
        "analyzed": n,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_conf, 4),
        "avg_time_seconds": round(avg_time, 3),
        "incidents": results_list,
    }
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    quality_path = out_dir / "quality_report.txt"
    quality_lines = [
        "Data Quality Report",
        "=" * 40,
        f"Total rows: {report.total_rows:,}",
        f"Unique incidents: {report.unique_incidents:,}",
        f"Success rate: {report.parsing_success_rate:.1%}",
        f"Parse time: {report.parse_time_seconds:.2f}s",
        f"Analyzable: {report.analyzable_incidents:,}",
        f"Non-technical: {report.excluded_non_technical:,}",
        "",
        "Field Completeness:",
    ]
    for fld, pct in sorted(report.field_completeness.items()):
        quality_lines.append(f"  {fld}: {pct:.1%}")
    quality_path.write_text("\n".join(quality_lines), encoding="utf-8")

    console.print(f"\n  Results   : {json_path}")
    console.print(f"  Quality   : {quality_path}")
    console.print()


# ── export-metrics ─────────────────────────────────────────────────


@cli.command("export-metrics")
@click.option(
    "--format",
    "fmt",
    default="prometheus",
    type=click.Choice(["prometheus"]),
    show_default=True,
    help="Metrics export format.",
)
@click.pass_context
def export_metrics(ctx: click.Context, fmt: str) -> None:
    """Export pipeline metrics in Prometheus text format.

    \b
    Example:
      python main.py export-metrics --format prometheus
    """
    try:
        from prometheus_client import generate_latest

        click.echo(generate_latest().decode("utf-8"))
    except ImportError:
        console.print(
            "[yellow]prometheus_client not installed — no metrics available.[/yellow]",
        )
        console.print("Install with: pip install prometheus_client")
        raise SystemExit(1)


# ── version ────────────────────────────────────────────────────────


@cli.command()
def version() -> None:
    """Show version and dependency information.

    \b
    Example:
      python main.py version
    """
    import platform

    console.print("[bold]Autonomous War-Room Simulator[/bold]")
    console.print(f"  Version  : 1.0.0")
    console.print(f"  Python   : {platform.python_version()}")

    deps = {
        "click": "click",
        "rich": "rich",
        "pydantic": "pydantic",
        "structlog": "structlog",
        "pyyaml": "yaml",
        "fastapi": "fastapi",
        "sqlalchemy": "sqlalchemy",
        "jinja2": "jinja2",
    }
    for label, mod in deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", getattr(m, "VERSION", "?"))
            console.print(f"  {label:12s}: {ver}")
        except ImportError:
            console.print(f"  {label:12s}: [dim]not installed[/dim]")


# ── entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
