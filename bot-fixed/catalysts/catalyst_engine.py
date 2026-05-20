"""
catalysts/catalyst_engine.py
Main catalyst detection orchestrator. Fetches news, classifies it,
and returns scored Catalyst objects for scanner integration.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from catalysts.catalyst_scoring import compute_final_catalyst_score
from catalysts.news_classifier import classify_multiple
from core.enums import CatalystCategory
from core.models import Catalyst
from market_data.base_provider import BaseDataProvider
from market_data.market_cache import MarketCache


class CatalystEngine:
    """
    Detects and scores catalysts for a list of symbols.
    Uses the data provider's news feed + keyword classification.
    Falls back gracefully if news is unavailable.
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        cache: MarketCache | None = None,
        lookback_hours: int = 18,
    ) -> None:
        self._provider = provider
        self._cache = cache or MarketCache()
        self._lookback_hours = lookback_hours
        self._catalyst_store: dict[str, Catalyst | None] = {}

    async def score_symbols(
        self, symbols: list[str], concurrency: int = 10
    ) -> dict[str, float]:
        """
        Process all symbols and return {symbol: catalyst_score}.
        Score is 0.0 for symbols with no detected catalyst.
        """
        if not self._provider.supports_news:
            logger.debug("Provider does not support news — all catalyst scores = 0")
            return {sym: 0.0 for sym in symbols}

        sem = asyncio.Semaphore(concurrency)
        scores: dict[str, float] = {}

        async def process_one(symbol: str) -> None:
            async with sem:
                catalyst = await self._detect_catalyst(symbol)
                self._catalyst_store[symbol] = catalyst
                scores[symbol] = catalyst.strength * catalyst.confidence if catalyst else 0.0

        await asyncio.gather(*[process_one(sym) for sym in symbols])
        return scores

    async def _detect_catalyst(self, symbol: str) -> Catalyst | None:
        """Fetch news and classify catalyst for a single symbol."""
        cache_key = f"catalyst:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            published_after = datetime.now(tz=timezone.utc) - timedelta(
                hours=self._lookback_hours
            )
            news_items = await self._provider.get_news(
                symbol, published_after=published_after, limit=5
            )
        except Exception as exc:
            logger.debug("News fetch failed for {}: {}", symbol, exc)
            await self._cache.set(cache_key, None, ttl_seconds=300)
            return None

        if not news_items:
            await self._cache.set(cache_key, None, ttl_seconds=300)
            return None

        catalyst = await self._classify_news_async(symbol, news_items)
        await self._cache.set(cache_key, catalyst, ttl_seconds=300)
        return catalyst

    async def _classify_news_async(
        self, symbol: str, news_items: list[dict[str, Any]]
    ) -> "Catalyst | None":
        """
        Classify news items with sentiment-aware scoring — v2.

        Critical improvement over v1: sentiment determines whether a catalyst
        is POSITIVE or NEGATIVE for price, not just what TYPE it is.
        "Lowers guidance" and "raises guidance" now produce different scores.
        A negative FDA event correctly gets low/zero confidence rather than 0.92.
        """
        from catalysts.catalyst_scoring import classify_news_timing
        from core.enums import NewsTimingCategory

        texts: list[str] = []
        published_at: datetime | None = None

        for item in news_items[:3]:
            title = item.get("title", "")
            description = item.get("description", "")
            if title:
                texts.append(title)
            if description:
                texts.append(description[:300])  # Cap body length for LLM cost
            if published_at is None:
                pub_str = item.get("published_utc", "")
                if pub_str:
                    try:
                        published_at = datetime.fromisoformat(
                            pub_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

        if not texts:
            return None

        # Sentiment-aware classification (v2 — key change from v1)
        result = await classify_multiple_with_sentiment(texts)

        if result.category == CatalystCategory.UNKNOWN and result.confidence < 0.1:
            return None

        # CRITICAL FILTER: negative sentiment catalysts are NOT trading setups.
        # A gap-up on bad news (earnings miss, trial failure) is a TRAP.
        # We set confidence to near-zero for bearish catalysts so they don't
        # score well and don't pass the catalyst confidence threshold.
        effective_confidence = result.combined_confidence or result.confidence
        if result.sentiment == NewsSentiment.NEGATIVE:
            # Bearish catalyst — severely penalise confidence
            effective_confidence *= 0.15
            logger.debug(
                "{}: NEGATIVE catalyst detected ({}) — confidence penalised to {:.2f}",
                symbol, result.category.value, effective_confidence,
            )
        elif result.sentiment == NewsSentiment.NEUTRAL:
            # Ambiguous — modest penalty
            effective_confidence *= 0.70

        # Sentiment score logged for explainability
        logger.debug(
            "{}: catalyst={} sentiment={} sent_score={:.2f} combined_conf={:.2f}",
            symbol, result.category.value, result.sentiment.value,
            result.sentiment_score, effective_confidence,
        )

        timing = classify_news_timing(published_at)
        strength = compute_final_catalyst_score(
            result.category, effective_confidence, published_at, timing
        )

        recency_hours = 0.0
        if published_at:
            recency_hours = (
                datetime.now(tz=timezone.utc) - published_at
            ).total_seconds() / 3600.0

        sentiment_tag = f"|{result.sentiment.value}" if result.sentiment != NewsSentiment.UNKNOWN else ""
        summary = (
            f"{result.category.value}{sentiment_tag} [{timing.value}]: "
            f"{', '.join(result.matched_keywords[:3])}"
        )

        return Catalyst(
            symbol=symbol,
            detected_at=datetime.now(tz=timezone.utc),
            category=result.category,
            confidence=round(effective_confidence, 3),
            strength=round(strength, 3),
            summary=summary,
            source="polygon_news",
            recency_hours=round(recency_hours, 1),
            news_timing=timing,
        )

    def get_catalyst(self, symbol: str) -> Catalyst | None:
        """Return the most recently detected catalyst for a symbol."""
        return self._catalyst_store.get(symbol)

    def clear(self) -> None:
        self._catalyst_store.clear()
