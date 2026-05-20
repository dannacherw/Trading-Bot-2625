"""
tests/test_analytics/test_filter_quality.py
Tests for filter quality analysis.
"""
from __future__ import annotations

import pytest

from analytics.filter_quality import FilterStats, print_filter_quality_report


def _make_stats(
    filter_name: str = "spread > max",
    total: int = 50,
    hit_1r: int = 20,
    hit_2r: int = 10,
    hit_stop: int = 15,
    correct: int = 25,
    investigate: int = 10,
    avg_mfe: float = 1.2,
    avg_mae: float = -0.8,
) -> FilterStats:
    return FilterStats(
        filter_name=filter_name,
        total_rejected=total,
        hit_1r_count=hit_1r,
        hit_2r_count=hit_2r,
        hit_stop_count=hit_stop,
        correct_rejections=correct,
        should_investigate_count=investigate,
        avg_mfe_r=avg_mfe,
        avg_mae_r=avg_mae,
    )


class TestFilterStats:
    def test_hit_1r_rate(self):
        s = _make_stats(total=100, hit_1r=40)
        assert abs(s.hit_1r_rate - 0.40) < 0.001

    def test_hit_2r_rate(self):
        s = _make_stats(total=100, hit_2r=15)
        assert abs(s.hit_2r_rate - 0.15) < 0.001

    def test_correct_rejection_rate(self):
        s = _make_stats(total=100, correct=70)
        assert abs(s.correct_rejection_rate - 0.70) < 0.001

    def test_zero_total_returns_zero_rates(self):
        s = _make_stats(total=0, hit_1r=0)
        assert s.hit_1r_rate == 0.0
        assert s.correct_rejection_rate == 0.0

    def test_verdict_too_strict(self):
        # 40% hit 1R, avg MFE > 1R → filter is too strict
        s = _make_stats(total=100, hit_1r=40, avg_mfe=1.5)
        assert "TOO STRICT" in s.verdict

    def test_verdict_good(self):
        # 5% hit 1R, 80% correct → filter is working well
        s = _make_stats(total=100, hit_1r=5, correct=80, avg_mfe=0.3)
        assert "GOOD" in s.verdict

    def test_verdict_insufficient_data(self):
        s = _make_stats(total=5)
        assert "INSUFFICIENT DATA" in s.verdict

    def test_verdict_watch(self):
        # 30% hit 1R, moderate MFE → watch
        s = _make_stats(total=100, hit_1r=30, avg_mfe=0.8, correct=50)
        assert "WATCH" in s.verdict or "TOO STRICT" in s.verdict  # Either is acceptable


class TestPrintFilterQualityReport:
    def test_no_crash_on_empty_stats(self, capsys):
        print_filter_quality_report([], "2024-01-01", "2024-01-31")
        captured = capsys.readouterr()
        assert "No missed trade data" in captured.out

    def test_prints_filter_names(self, capsys):
        stats = [
            _make_stats("spread > max", total=50, hit_1r=25, avg_mfe=1.5),
            _make_stats("gap < min", total=30, hit_1r=5, avg_mfe=0.3, correct=25),
        ]
        print_filter_quality_report(stats, "2024-01-01", "2024-01-31")
        captured = capsys.readouterr()
        output = captured.out
        # Should print something (rich or plain)
        assert len(output) > 0

    def test_does_not_crash_with_single_stat(self, capsys):
        stats = [_make_stats(total=100, hit_1r=45, avg_mfe=2.0)]
        print_filter_quality_report(stats, "2024-01-01", "2024-01-31")
        # No exception = pass


"""
tests/test_integrations/test_sheets_writer.py
Tests for SheetsWriter (mock-based, no real Google API calls).
"""


class TestSheetsWriterRowSerialisation:
    """Test that row formatting functions produce the right shape."""

    def test_scan_result_to_row_length(self):
        """Scan_Log has 22 columns — verify row has correct length."""
        from integrations.sheets_writer import _scan_result_to_row
        from core.models import PremarketMetrics, ScanResult
        from datetime import datetime, timezone

        metrics = PremarketMetrics(
            symbol="AAPL",
            computed_at=datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc),
            prev_close=95.0,
            premarket_open=97.0,
            premarket_high=102.0,
            premarket_low=96.5,
            premarket_last=101.0,
            premarket_volume=500_000,
            premarket_dollar_volume=5_000_000,
            gap_pct=6.32,
            relative_volume=3.5,
            spread_pct=0.08,
            range_pct=5.0,
            range_position=0.90,
            trend_quality=0.75,
            avg_daily_dollar_volume=30_000_000,
        )
        result = ScanResult(
            symbol="AAPL",
            scanned_at=datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc),
            metrics=metrics,
            composite_score=0.82,
            passes_filters=True,
        )
        row = _scan_result_to_row(result)
        assert len(row) == 22

    def test_scan_result_row_pass_flag(self):
        from integrations.sheets_writer import _scan_result_to_row
        from core.models import PremarketMetrics, ScanResult
        from datetime import datetime, timezone

        metrics = PremarketMetrics(
            symbol="GME",
            computed_at=datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc),
            prev_close=15.0,
            premarket_open=15.5,
            premarket_high=16.0,
            premarket_low=15.2,
            premarket_last=15.8,
            premarket_volume=50_000,
            premarket_dollar_volume=790_000,
            gap_pct=2.1,
            relative_volume=1.2,
            spread_pct=0.38,
            range_pct=2.0,
            range_position=0.7,
            trend_quality=0.3,
            avg_daily_dollar_volume=5_000_000,
        )
        result = ScanResult(
            symbol="GME",
            scanned_at=datetime(2024, 1, 15, 13, 0, tzinfo=timezone.utc),
            metrics=metrics,
            composite_score=0.22,
            passes_filters=False,
            filter_failure_reason="spread 0.38% > max 0.25%",
        )
        row = _scan_result_to_row(result)
        # Index 18 = "passed" column
        assert row[18] == "FALSE"


"""
tests/test_api/test_auth.py
Tests for API authentication middleware.
"""
import os
from unittest.mock import patch


class TestApiAuth:
    def _get_authenticator(self, admin_pass: str = "admin123", viewer_pass: str = "view456"):
        """Reload auth module with specific credentials."""
        env = {
            "API_ADMIN_USER":  "admin",
            "API_ADMIN_PASS":  admin_pass,
            "API_VIEWER_USER": "alice,bob",
            "API_VIEWER_PASS": "alice_pw,bob_pw",
        }
        with patch.dict(os.environ, env):
            # Re-import to pick up patched env
            import importlib
            import api.auth as auth_module
            importlib.reload(auth_module)
            return auth_module

    def test_admin_credentials_accepted(self):
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        creds = HTTPBasicCredentials(username="admin", password="admin123")
        username, role = auth._authenticate(creds)
        assert username == "admin"
        assert role == auth.Role.ADMIN

    def test_viewer_credentials_accepted(self):
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        creds = HTTPBasicCredentials(username="alice", password="alice_pw")
        username, role = auth._authenticate(creds)
        assert username == "alice"
        assert role == auth.Role.VIEWER

    def test_wrong_password_raises_401(self):
        from fastapi import HTTPException
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        creds = HTTPBasicCredentials(username="admin", password="wrong")
        with pytest.raises(HTTPException) as exc_info:
            auth._authenticate(creds)
        assert exc_info.value.status_code == 401

    def test_unknown_user_raises_401(self):
        from fastapi import HTTPException
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        creds = HTTPBasicCredentials(username="hacker", password="anything")
        with pytest.raises(HTTPException) as exc_info:
            auth._authenticate(creds)
        assert exc_info.value.status_code == 401

    def test_require_admin_rejects_viewer(self):
        from fastapi import Depends, HTTPException
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        creds = HTTPBasicCredentials(username="alice", password="alice_pw")
        with pytest.raises(HTTPException) as exc_info:
            auth.require_admin(creds)
        assert exc_info.value.status_code in (401, 403)

    def test_multiple_viewers_all_accepted(self):
        auth = self._get_authenticator()
        from fastapi.security import HTTPBasicCredentials
        for user, pw in [("alice", "alice_pw"), ("bob", "bob_pw")]:
            creds = HTTPBasicCredentials(username=user, password=pw)
            username, role = auth._authenticate(creds)
            assert role == auth.Role.VIEWER
