"""CLI helpers — argument validation, output formatting, Rich widgets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console(stderr=True)

# Scenario metadata for display in ``list-scenarios``.
SCENARIO_META: Dict[str, Dict[str, str]] = {
    "memory_leak": {
        "description": "Gradual heap exhaustion on the payment-service",
        "severity": "P2 MEDIUM",
        "duration": "~25 min",
    },
    "cpu_spike": {
        "description": "Sudden CPU saturation on the fraud scoring engine",
        "severity": "P1 HIGH",
        "duration": "~10 min",
    },
    "database_timeout": {
        "description": "Primary database becomes unresponsive",
        "severity": "P0 CRITICAL",
        "duration": "~12 min",
    },
    "network_latency": {
        "description": "Upstream network degradation at the API gateway",
        "severity": "P1 HIGH",
        "duration": "~8 min",
    },
}


# ── validation helpers ─────────────────────────────────────────────


def validate_scenario(scenario: str, available: List[str]) -> bool:
    """Return ``True`` if *scenario* is in *available*."""
    return scenario in available


def validate_format(fmt: str) -> bool:
    """Return ``True`` if *fmt* is a supported report format."""
    return fmt.lower() in {"html", "markdown", "json", "pdf"}


# ── formatting helpers ─────────────────────────────────────────────


def format_duration(seconds: float) -> str:
    """Format *seconds* as ``2m 34s`` or ``1.2s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_confidence(confidence: float) -> str:
    """Format *confidence* (0.0-1.0) as a coloured percentage."""
    pct = confidence * 100
    if pct >= 85:
        return f"[green]{pct:.0f}%[/green]"
    if pct >= 60:
        return f"[yellow]{pct:.0f}%[/yellow]"
    return f"[red]{pct:.0f}%[/red]"


def format_file_size(size_bytes: int) -> str:
    """Format *size_bytes* as ``1.2 MB`` / ``845 KB`` etc."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


# ── Rich widgets ───────────────────────────────────────────────────


def create_progress() -> Progress:
    """Create a Rich Progress bar suitable for pipeline stages."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def display_result_panel(result: Dict[str, Any]) -> None:
    """Display a success panel summarising a pipeline run."""
    root_cause = result.get("root_cause", "unknown")
    confidence = result.get("confidence", 0.0)
    report_paths = result.get("report_paths", {})
    duration = result.get("duration", 0.0)
    scenario = result.get("scenario", "")

    files_display = ", ".join(str(p) for p in report_paths.values()) or "none"

    body = (
        f"[green]✅ Incident Analysis Complete[/green]\n\n"
        f"Scenario   : [cyan]{scenario}[/cyan]\n"
        f"Root Cause : {root_cause}  ({format_confidence(confidence)})\n"
        f"Reports    : {files_display}\n"
        f"Duration   : {format_duration(duration)}"
    )

    console.print(
        Panel(body, title="Result", border_style="green", padding=(1, 2)),
    )


def display_error(error: Exception, context: str = "") -> None:
    """Display a formatted error panel."""
    msg = f"[red]✗ Error{f' ({context})' if context else ''}[/red]\n\n{error}"
    console.print(Panel(msg, title="Error", border_style="red", padding=(1, 2)))


def display_scenarios_table(scenarios: List[str]) -> None:
    """Print a Rich table of available scenarios."""
    table = Table(
        title="Available Incident Scenarios",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Severity", style="yellow")
    table.add_column("Duration", style="green")

    for name in scenarios:
        meta = SCENARIO_META.get(name, {})
        table.add_row(
            name,
            meta.get("description", "—"),
            meta.get("severity", "—"),
            meta.get("duration", "—"),
        )

    console.print(table)


def display_analytics_summary(analytics: Dict[str, Any]) -> None:
    """Print a Rich summary of historical analytics."""
    console.print("\n[bold]📊 Historical Analysis[/bold]\n")
    console.print(f"  Total Incidents  : {analytics.get('total_incidents', 0)}")
    console.print(f"  Average MTTR     : {analytics.get('mttr', 0):.1f} minutes")
    console.print(f"  Average MTTD     : {analytics.get('mttd', 0):.1f} minutes")
    console.print(
        f"  SLO Compliance   : {analytics.get('slo_compliance', 0):.0%}",
    )
    console.print(f"  Total Cost       : ${analytics.get('total_cost', 0):.2f}\n")

    causes = analytics.get("common_root_causes", [])
    if causes:
        table = Table(title="Top Root Causes", show_header=True)
        table.add_column("Root Cause", style="cyan")
        table.add_column("Count", style="magenta")
        for cause, count in causes[:5]:
            table.add_row(str(cause), str(count))
        console.print(table)


def confirm(message: str, default: bool = True) -> bool:
    """Ask the user for yes/no confirmation."""
    suffix = " [Y/n] " if default else " [y/N] "
    resp = console.input(f"[bold]{message}{suffix}[/bold]").strip().lower()
    if not resp:
        return default
    return resp in ("y", "yes")
