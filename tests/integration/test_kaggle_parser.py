"""Tests for the Kaggle IT incident parser.

Covers: full-dataset parsing, event-log grouping, missing data handling,
priority extraction, synthetic log generation, ground truth mapping,
inquiry filtering, and data quality metrics.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from integrations.real_data.kaggle_parser import (
    KaggleIncidentParser,
    KaggleIncident,
    SyntheticLogEntry,
    DataQualityReport,
)


# ── Fixtures ───────────────────────────────────────────────────────

REAL_CSV = Path(__file__).resolve().parents[2] / "data" / "real" / "incident_event_log.csv"
REAL_CSV_EXISTS = REAL_CSV.exists()


def _make_csv(rows: list[list[str]], tmp_path: Path) -> str:
    """Write a minimal CSV for unit-level tests."""
    header = (
        "number,incident_state,active,reassignment_count,reopen_count,"
        "sys_mod_count,made_sla,caller_id,opened_by,opened_at,"
        "sys_created_by,sys_created_at,sys_updated_by,sys_updated_at,"
        "contact_type,location,category,subcategory,u_symptom,cmdb_ci,"
        "impact,urgency,priority,assignment_group,assigned_to,knowledge,"
        "u_priority_confirmation,notify,problem_id,rfc,vendor,caused_by,"
        "closed_code,resolved_by,resolved_at,closed_at"
    )
    lines = [header]
    for row in rows:
        lines.append(",".join(row))
    csv_path = tmp_path / "test_incidents.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return str(csv_path)


def _row(
    number: str = "INC0000001",
    state: str = "New",
    opened_at: str = "1/1/2016 08:00",
    resolved_at: str = "1/1/2016 10:00",
    closed_at: str = "1/1/2016 12:00",
    category: str = "Category 26",
    subcategory: str = "Subcategory 174",
    symptom: str = "Symptom 491",
    priority: str = "3 - Moderate",
    impact: str = "2 - Medium",
    urgency: str = "2 - Medium",
    assignment_group: str = "Group 56",
    closed_code: str = "code 6",
    reopen_count: str = "0",
) -> list[str]:
    """Build a single CSV row with sensible defaults."""
    return [
        number, state, "true", "0", reopen_count,
        "1", "true", "Caller_1", "Opener_1", opened_at,
        "Creator_1", opened_at, "Updater_1", opened_at,
        "Phone", "Location_1", category, subcategory, symptom, "CI_1",
        impact, urgency, priority, assignment_group, "Assigned_1", "false",
        "false", "1", "", "", "", "",
        closed_code, "Resolver_1", resolved_at, closed_at,
    ]


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Full dataset parsing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not REAL_CSV_EXISTS, reason="Real CSV not present")
class TestParseFullDataset:
    def test_parses_all_unique_incidents(self):
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(str(REAL_CSV))
        assert len(incidents) == 24918, f"Expected 24918, got {len(incidents)}"

    def test_all_incidents_have_required_fields(self):
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(str(REAL_CSV))
        for inc in incidents[:100]:  # spot-check first 100
            assert inc.incident_id, "incident_id must not be empty"
            assert inc.opened_at is not None, "opened_at must not be None"
            assert inc.domain_category in (
                "database", "network", "hardware", "software",
                "application", "inquiry",
            )
            assert 1 <= inc.priority_num <= 4
            assert inc.severity in ("CRITICAL", "HIGH", "MODERATE", "LOW")

    def test_parsing_completes_within_time_limit(self):
        parser = KaggleIncidentParser()
        t0 = time.perf_counter()
        parser.parse_file(str(REAL_CSV))
        elapsed = time.perf_counter() - t0
        assert elapsed < 20.0, f"Parsing took {elapsed:.1f}s (limit: 20s)"

    def test_no_duplicate_incidents(self):
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(str(REAL_CSV))
        ids = [inc.incident_id for inc in incidents]
        assert len(ids) == len(set(ids)), "Duplicate incident IDs detected"


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Event log grouping
# ═══════════════════════════════════════════════════════════════════

class TestEventLogGrouping:
    def test_multiple_events_group_to_single_incident(self, tmp_path):
        rows = [
            _row("INC0000042", "New", "15/3/2016 08:00", "?", "?",
                 closed_code="?"),
            _row("INC0000042", "Active", "15/3/2016 08:00", "?", "?",
                 assignment_group="Group 10", closed_code="?"),
            _row("INC0000042", "Active", "15/3/2016 08:00", "?", "?",
                 assignment_group="Group 10", closed_code="?"),
            _row("INC0000042", "Resolved", "15/3/2016 08:00",
                 "15/3/2016 14:00", "?", closed_code="?"),
            _row("INC0000042", "Closed", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00",
                 closed_code="code 5"),
        ]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.incident_id == "INC0000042"
        assert inc.event_count == 5

    def test_opened_at_uses_first_event(self, tmp_path):
        rows = [
            _row("INC0000042", "New", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00"),
            _row("INC0000042", "Closed", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00"),
        ]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert incidents[0].opened_at == datetime(2016, 3, 15, 8, 0)

    def test_closed_at_uses_last_event(self, tmp_path):
        rows = [
            _row("INC0000042", "New", "15/3/2016 08:00", "?", "?",
                 closed_code="?"),
            _row("INC0000042", "Closed", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00",
                 closed_code="code 5"),
        ]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert incidents[0].closed_at == datetime(2016, 3, 15, 18, 0)
        assert incidents[0].closed_code == "code 5"

    def test_assignment_group_uses_last_event(self, tmp_path):
        rows = [
            _row("INC0000042", "New", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00",
                 assignment_group="Group 1"),
            _row("INC0000042", "Closed", "15/3/2016 08:00",
                 "15/3/2016 14:00", "15/3/2016 18:00",
                 assignment_group="Group 99"),
        ]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert incidents[0].assignment_group == "Group 99"


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Missing data handling
# ═══════════════════════════════════════════════════════════════════

class TestMissingDataHandling:
    def test_symptom_fallback_to_subcategory(self, tmp_path):
        rows = [_row(symptom="?", subcategory="Subcategory 42")]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert "Subcategory 42" in incidents[0].symptom

    def test_symptom_fallback_to_category(self, tmp_path):
        rows = [_row(symptom="?", subcategory="?")]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert "Unknown" in incidents[0].symptom
        assert "Category" in incidents[0].symptom

    def test_resolved_at_none_uses_closed_at(self, tmp_path):
        rows = [_row(resolved_at="?", closed_at="2/1/2016 14:00")]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        # resolved_at should fall back to closed_at
        assert incidents[0].resolved_at == datetime(2016, 1, 2, 14, 0)

    def test_both_timestamps_missing_yields_unresolved(self, tmp_path):
        rows = [_row(resolved_at="?", closed_at="?")]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)

        assert incidents[0].resolved_at is None
        assert incidents[0].closed_at is None


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Priority extraction
# ═══════════════════════════════════════════════════════════════════

class TestPriorityExtraction:
    @pytest.mark.parametrize(
        "raw, expected_num, expected_sev",
        [
            ("1 - Critical", 1, "CRITICAL"),
            ("2 - High", 2, "HIGH"),
            ("3 - Moderate", 3, "MODERATE"),
            ("4 - Low", 4, "LOW"),
            ("2-High", 2, "HIGH"),             # no space
            ("P3", 3, "MODERATE"),             # P-prefix
            ("High", 2, "HIGH"),               # name only
            ("", 3, "MODERATE"),               # empty → default
            ("nan", 3, "MODERATE"),            # NaN → default
            ("5", 4, "LOW"),                   # out of range → clamped
            ("0", 1, "CRITICAL"),              # out of range → clamped
            ("2", 2, "HIGH"),                  # number only
        ],
    )
    def test_priority_parsing(self, raw, expected_num, expected_sev):
        num, sev = KaggleIncidentParser._extract_priority(raw)
        assert num == expected_num, f"{raw!r} → {num}, expected {expected_num}"
        assert sev == expected_sev, f"{raw!r} → {sev}, expected {expected_sev}"


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Synthetic log generation
# ═══════════════════════════════════════════════════════════════════

class TestSyntheticLogGeneration:
    def test_generates_3_to_7_logs_per_incident(self, tmp_path):
        rows = [_row()]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        assert 3 <= len(logs) <= 7

    def test_all_logs_share_correlation_id(self, tmp_path):
        rows = [_row(number="INC0000099")]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        for log in logs:
            assert log.correlation_id == "INC0000099"

    def test_logs_contain_domain_specific_content(self, tmp_path):
        # We test a database-mapped category
        parser = KaggleIncidentParser()
        # Find a category that maps to "database"
        db_cat = None
        for i in range(200):
            cat = f"Category {i}"
            if parser._map_to_domain(cat) == "database":
                db_cat = cat
                break
        if db_cat is None:
            pytest.skip("No category maps to database in this hash space")

        rows = [_row(category=db_cat, priority="1 - Critical")]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        messages = " ".join(lg.message for lg in logs)
        # Database logs should contain DB-related keywords
        db_keywords = ["database", "connection", "query", "pool",
                       "deadlock", "transaction", "replica", "vacuum"]
        assert any(kw.lower() in messages.lower() for kw in db_keywords), \
            f"No DB keywords found in: {messages}"

    def test_log_levels_appropriate_for_priority(self, tmp_path):
        # High priority → should have ERROR logs
        # Use a category that maps to a technical domain (not inquiry)
        parser = KaggleIncidentParser()
        tech_cat = None
        for i in range(200):
            cat = f"Category {i}"
            if parser._map_to_domain(cat) != "inquiry":
                tech_cat = cat
                break
        assert tech_cat is not None, "Could not find a technical category"

        rows = [_row(priority="1 - Critical", category=tech_cat)]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        levels = {lg.level for lg in logs}
        assert "ERROR" in levels, f"P1 incident should have ERROR logs, got {levels}"

    def test_logs_chronologically_ordered(self, tmp_path):
        rows = [_row()]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        timestamps = [lg.timestamp for lg in logs]
        assert timestamps == sorted(timestamps), "Logs must be chronologically sorted"


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Ground truth mapping
# ═══════════════════════════════════════════════════════════════════

class TestGroundTruthMapping:
    def test_maps_domains_to_root_causes(self, tmp_path):
        parser = KaggleIncidentParser()
        # Find categories for each domain
        domain_cats = {}
        for i in range(200):
            cat = f"Category {i}"
            domain = parser._map_to_domain(cat)
            if domain not in domain_cats and domain != "inquiry":
                domain_cats[domain] = cat

        rows = []
        for domain, cat in domain_cats.items():
            rows.append(_row(
                number=f"INC_{domain}",
                category=cat,
            ))
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        gt = parser.get_ground_truth(incidents)

        for inc in incidents:
            entry = gt[inc.incident_id]
            assert entry.primary_cause != "", f"No primary cause for {inc.domain_category}"
            assert len(entry.valid_answers) >= 2, \
                f"Should have multiple valid answers for {inc.domain_category}"

    def test_fuzzy_matching_allows_variants(self, tmp_path):
        parser = KaggleIncidentParser()
        # Find a database category
        db_cat = None
        for i in range(200):
            if parser._map_to_domain(f"Category {i}") == "database":
                db_cat = f"Category {i}"
                break

        if db_cat is None:
            pytest.skip("No DB category found")

        rows = [_row(number="INC_DB1", category=db_cat)]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        gt = parser.get_ground_truth(incidents)

        entry = gt["INC_DB1"]
        # All these should be valid
        assert "database_timeout" in entry.valid_answers
        assert "database_issue" in entry.valid_answers


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Inquiry / Help filtering
# ═══════════════════════════════════════════════════════════════════

class TestInquiryFiltering:
    def test_inquiry_incidents_marked_non_technical(self, tmp_path):
        parser = KaggleIncidentParser()
        inq_cat = None
        for i in range(200):
            if parser._map_to_domain(f"Category {i}") == "inquiry":
                inq_cat = f"Category {i}"
                break
        if inq_cat is None:
            pytest.skip("No inquiry category found")

        rows = [_row(number="INC_INQ1", category=inq_cat)]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)

        assert len(incidents) == 1
        assert incidents[0].is_technical is False

    def test_excluded_from_analyzable_count(self, tmp_path):
        parser = KaggleIncidentParser()
        inq_cat = None
        tech_cat = None
        for i in range(200):
            domain = parser._map_to_domain(f"Category {i}")
            if domain == "inquiry" and inq_cat is None:
                inq_cat = f"Category {i}"
            elif domain != "inquiry" and tech_cat is None:
                tech_cat = f"Category {i}"

        if inq_cat is None:
            pytest.skip("No inquiry category found")

        rows = [
            _row(number="INC_TECH1", category=tech_cat),
            _row(number="INC_INQ1", category=inq_cat),
            _row(number="INC_TECH2", category=tech_cat),
        ]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        report = parser.generate_data_quality_report()

        assert report.excluded_non_technical == 1
        assert report.analyzable_incidents == 2

    def test_inquiry_logs_are_info_level(self, tmp_path):
        parser = KaggleIncidentParser()
        inq_cat = None
        for i in range(200):
            if parser._map_to_domain(f"Category {i}") == "inquiry":
                inq_cat = f"Category {i}"
                break
        if inq_cat is None:
            pytest.skip("No inquiry category found")

        rows = [_row(number="INC_INQ1", category=inq_cat, priority="4 - Low")]
        csv = _make_csv(rows, tmp_path)
        incidents = parser.parse_file(csv)
        logs = parser.convert_to_logs(incidents)

        # Inquiry logs should all be INFO (non-technical)
        for lg in logs:
            assert lg.level == "INFO", f"Inquiry log should be INFO, got {lg.level}"


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Data quality metrics
# ═══════════════════════════════════════════════════════════════════

class TestDataQualityMetrics:
    def test_report_has_all_required_sections(self, tmp_path):
        rows = [
            _row(number="INC0000001"),
            _row(number="INC0000001", state="Closed"),
            _row(number="INC0000002"),
        ]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        parser.parse_file(csv)
        report = parser.generate_data_quality_report()

        assert report.total_rows == 3
        assert report.unique_incidents >= 2
        assert report.parsing_success_rate > 0
        assert isinstance(report.field_completeness, dict)
        assert isinstance(report.category_distribution, dict)
        assert isinstance(report.priority_distribution, dict)
        assert isinstance(report.state_distribution, dict)
        assert report.analyzable_incidents >= 0
        assert report.parse_time_seconds >= 0

    def test_field_completeness_calculated(self, tmp_path):
        rows = [_row()]
        csv = _make_csv(rows, tmp_path)
        parser = KaggleIncidentParser()
        parser.parse_file(csv)
        report = parser.generate_data_quality_report()

        assert "number" in report.field_completeness
        assert "priority" in report.field_completeness
        assert "category" in report.field_completeness
        # All fields present → completeness should be 1.0
        assert report.field_completeness["number"] == 1.0

    @pytest.mark.skipif(not REAL_CSV_EXISTS, reason="Real CSV not present")
    def test_success_rate_above_threshold(self):
        parser = KaggleIncidentParser()
        parser.parse_file(str(REAL_CSV))
        report = parser.generate_data_quality_report()

        assert report.parsing_success_rate >= 0.95, \
            f"Success rate {report.parsing_success_rate:.1%} < 95%"

    @pytest.mark.skipif(not REAL_CSV_EXISTS, reason="Real CSV not present")
    def test_full_report_quality(self):
        parser = KaggleIncidentParser()
        parser.parse_file(str(REAL_CSV))
        report = parser.generate_data_quality_report()

        assert report.total_rows == 141712
        assert report.unique_incidents == 24918
        assert report.avg_events_per_incident > 1.0
        assert len(report.category_distribution) >= 2
        assert len(report.priority_distribution) >= 2
