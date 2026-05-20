"""
tests/test_scanner/test_catalyst.py
Tests for news classification and catalyst scoring.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from catalysts.catalyst_scoring import (
    compute_catalyst_strength,
    compute_final_catalyst_score,
    compute_recency_decay,
)
from catalysts.news_classifier import classify_headline, classify_multiple
from core.enums import CatalystCategory


class TestNewsClassifier:
    def test_classifies_earnings(self):
        result = classify_headline("Company beats earnings estimates, EPS of $2.50 vs $2.20 expected")
        assert result.category == CatalystCategory.EARNINGS
        assert result.confidence > 0.7

    def test_classifies_fda_news(self):
        result = classify_headline("FDA grants accelerated approval for Phase 3 drug trial results")
        assert result.category == CatalystCategory.FDA_OR_BIOTECH_NEWS
        assert result.confidence > 0.8

    def test_classifies_analyst_upgrade(self):
        result = classify_headline("Goldman Sachs upgrades stock to Buy, raises price target to $150")
        assert result.category == CatalystCategory.ANALYST_UPGRADE
        assert result.confidence > 0.7

    def test_classifies_ma(self):
        result = classify_headline("Company to be acquired by larger rival in $5B buyout deal")
        assert result.category == CatalystCategory.M_AND_A
        assert result.confidence > 0.7

    def test_unknown_headline(self):
        result = classify_headline("random unrelated text about something else entirely")
        # May classify as UNKNOWN or with low confidence
        assert result.confidence <= 0.6 or result.category == CatalystCategory.UNKNOWN

    def test_classify_multiple_takes_highest_confidence(self):
        texts = [
            "Company reports quarterly results",  # lower confidence
            "FDA approves breakthrough therapy drug for cancer",  # higher confidence
        ]
        result = classify_multiple(texts)
        assert result.category == CatalystCategory.FDA_OR_BIOTECH_NEWS

    def test_empty_list(self):
        result = classify_multiple([])
        assert result.category == CatalystCategory.UNKNOWN

    def test_matched_keywords_populated(self):
        result = classify_headline("EPS beats earnings estimates significantly")
        assert len(result.matched_keywords) > 0

    def test_case_insensitive(self):
        result1 = classify_headline("FDA APPROVES NEW DRUG")
        result2 = classify_headline("fda approves new drug")
        assert result1.category == result2.category


class TestCatalystScoring:
    def test_strength_proportional_to_category(self):
        fda = compute_catalyst_strength(CatalystCategory.FDA_OR_BIOTECH_NEWS, 0.9)
        press = compute_catalyst_strength(CatalystCategory.PRESS_RELEASE, 0.9)
        assert fda > press

    def test_strength_proportional_to_confidence(self):
        high = compute_catalyst_strength(CatalystCategory.EARNINGS, 0.9)
        low = compute_catalyst_strength(CatalystCategory.EARNINGS, 0.3)
        assert high > low

    def test_strength_capped_at_1(self):
        s = compute_catalyst_strength(CatalystCategory.EARNINGS, 1.0)
        assert s <= 1.0

    def test_recency_decay_within_2h(self):
        now = datetime.now(tz=timezone.utc)
        published = now - timedelta(hours=1)
        decay = compute_recency_decay(published, now)
        assert decay == pytest.approx(1.0)

    def test_recency_decay_at_24h(self):
        now = datetime.now(tz=timezone.utc)
        published = now - timedelta(hours=24)
        decay = compute_recency_decay(published, now)
        assert decay == pytest.approx(0.1, abs=0.05)

    def test_recency_decay_between_2_24h(self):
        now = datetime.now(tz=timezone.utc)
        published = now - timedelta(hours=12)
        decay = compute_recency_decay(published, now)
        assert 0.1 < decay < 1.0

    def test_final_score_without_time(self):
        score = compute_final_catalyst_score(CatalystCategory.EARNINGS, 0.9)
        assert 0.0 < score <= 1.0

    def test_final_score_with_old_news_lower(self):
        now = datetime.now(tz=timezone.utc)
        fresh = compute_final_catalyst_score(CatalystCategory.EARNINGS, 0.9,
                                             published_at=now - timedelta(hours=1))
        stale = compute_final_catalyst_score(CatalystCategory.EARNINGS, 0.9,
                                             published_at=now - timedelta(hours=20))
        assert fresh > stale
