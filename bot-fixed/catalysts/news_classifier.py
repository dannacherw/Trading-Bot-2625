"""
catalysts/news_classifier.py
Sentiment-aware news classifier — v2.

Replaces the v1 keyword-only classifier that had a critical flaw:
"Company X lowers guidance" and "Company X raises guidance" produced
identical scores because only the category keyword matched, not the
direction (positive/negative).

v2 architecture — two-stage pipeline:
  Stage 1: Category detection (same keyword approach as v1 — fast, reliable)
  Stage 2: Sentiment scoring — determines whether the catalyst is
            POSITIVE, NEGATIVE, or NEUTRAL for the stock price.

Sentiment is what actually drives win rate. A negative earnings catalyst
(miss + lowered guidance) that gaps up 5% is a trap — not a setup.
A positive catalyst with strong sentiment is the real edge.

Sentiment is computed by:
  1. Positive signal phrases (+1): beat, exceed, record, raises, strong, above
  2. Negative signal phrases (-1): miss, below, cut, lower, disappoints, warns
  3. Negation detection: "did not beat", "no longer raising" inverts signal
  4. Magnitude modifiers: "significantly", "materially" amplify the signal
  5. Conflict resolution: mixed signals → NEUTRAL with reduced confidence

OpenAI integration (optional, used when OPENAI_API_KEY is set):
  When available, stage 2 uses gpt-3.5-turbo to classify sentiment.
  Cost: ~$0.001 per headline (negligible).
  When unavailable, falls back to the rule-based stage 2 — still far
  better than v1's zero sentiment awareness.

The final catalyst confidence is a product of:
  category_confidence × sentiment_confidence × timing_multiplier
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from core.enums import CatalystCategory


# ---------------------------------------------------------------------------
# Sentiment enum
# ---------------------------------------------------------------------------

class NewsSentiment(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL  = "NEUTRAL"
    UNKNOWN  = "UNKNOWN"


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    category:        CatalystCategory
    confidence:      float
    matched_keywords: list[str]
    sentiment:       NewsSentiment = NewsSentiment.UNKNOWN
    sentiment_score: float = 0.0    # -1.0 (bearish) → +1.0 (bullish)
    sentiment_confidence: float = 0.0
    combined_confidence: float = 0.0  # category_conf × sentiment_conf

    @property
    def is_bullish(self) -> bool:
        return self.sentiment == NewsSentiment.POSITIVE

    @property
    def is_bearish(self) -> bool:
        return self.sentiment == NewsSentiment.NEGATIVE


# ---------------------------------------------------------------------------
# Category keyword mappings (unchanged from v1 — reliable)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: list[tuple[CatalystCategory, list[str], float]] = [
    (CatalystCategory.EARNINGS, [
        "earnings", "eps", "quarterly results", "q1 results", "q2 results",
        "q3 results", "q4 results", "beats estimates", "misses estimates",
        "profit", "revenue beat", "revenue miss", "annual results",
        "quarterly earnings", "per share",
    ], 0.90),

    (CatalystCategory.EARNINGS_GUIDANCE, [
        "guidance", "outlook", "forecast", "raises guidance", "lowers guidance",
        "updates guidance", "full-year outlook", "full year guidance",
        "raises outlook", "cuts outlook",
    ], 0.85),

    (CatalystCategory.FDA_OR_BIOTECH_NEWS, [
        "fda", "clinical trial", "phase 1", "phase 2", "phase 3",
        "drug approval", "nda", "bla", "pdufa", "inda", "biotech",
        "efficacy", "safety data", "trial results", "placebo",
        "accelerated approval", "breakthrough therapy", "complete response letter",
        "fast track", "orphan drug",
    ], 0.92),

    (CatalystCategory.ANALYST_UPGRADE, [
        "upgrade", "upgraded to buy", "raises price target", "raises pt",
        "outperform", "strong buy", "initiates coverage", "overweight",
        "price target increase", "upgrades to", "initiates with buy",
    ], 0.85),

    (CatalystCategory.ANALYST_DOWNGRADE, [
        "downgrade", "downgraded to", "lowers price target", "cuts pt",
        "underperform", "sell rating", "underweight",
        "price target decrease", "price target cut", "downgrades to",
    ], 0.85),

    (CatalystCategory.M_AND_A, [
        "merger", "acquisition", "acquires", "buyout", "takeover",
        "deal", "acquired by", "to be acquired",
        "strategic review", "going private", "tender offer",
    ], 0.90),

    (CatalystCategory.PARTNERSHIP_OR_CONTRACT, [
        "partnership", "collaboration", "agreement", "contract",
        "supply agreement", "licensing", "joint venture", "strategic alliance",
        "memo of understanding", "mou", "distribution agreement",
        "commercial agreement",
    ], 0.80),

    (CatalystCategory.PRODUCT_LAUNCH, [
        "launches", "launch", "unveils", "announces new product",
        "new product", "product release", "commercial launch", "begins shipping",
    ], 0.75),

    (CatalystCategory.LEGAL_OR_REGULATORY, [
        "lawsuit", "litigation", "settlement", "sec investigation",
        "regulatory approval", "regulatory clearance", "subpoena",
        "class action", "doj", "ftc", "antitrust",
    ], 0.80),

    (CatalystCategory.MACRO_RELATED, [
        "fed", "federal reserve", "interest rate", "inflation", "cpi",
        "jobs report", "gdp", "tariff", "trade war",
        "recession", "rate hike", "rate cut",
    ], 0.70),

    (CatalystCategory.PRESS_RELEASE, [
        "press release", "announces", "reports", "announces results",
    ], 0.50),
]


# ---------------------------------------------------------------------------
# Sentiment signal phrases
# ---------------------------------------------------------------------------

# Each entry: (phrase, base_score)
# score > 0 = bullish, < 0 = bearish
_POSITIVE_SIGNALS: list[tuple[str, float]] = [
    # Strong positive
    ("beats",            0.9), ("beat",              0.9),
    ("exceeds",          0.9), ("exceed",             0.9),
    ("surpasses",        0.9), ("surpass",            0.9),
    ("record",           0.8), ("record high",        0.95),
    ("raises guidance",  0.95), ("raises outlook",    0.95),
    ("raises forecast",  0.90), ("raises price target", 0.80),
    ("above expectations", 0.90), ("above estimates",  0.90),
    ("better than expected", 0.90),
    ("strong results",   0.85), ("strong revenue",    0.85),
    ("strong earnings",  0.85),
    ("fda approves",     0.95), ("fda approved",      0.95),
    ("approval",         0.75), ("approved",          0.80),
    ("positive data",    0.85), ("positive results",  0.85),
    ("successful trial", 0.90),
    ("upgrade",          0.75), ("upgraded",          0.75),
    ("outperform",       0.70),
    ("acquires",         0.65), ("acquired by",       0.70),
    ("buyout",           0.80), ("takeover",          0.75),
    ("partnership",      0.60), ("collaboration",     0.60),
    ("launches",         0.55), ("launch",            0.55),
    ("expands",          0.55), ("growth",            0.55),
    ("increases",        0.55), ("higher",            0.50),
    ("profit",           0.65), ("revenue growth",    0.70),
]

_NEGATIVE_SIGNALS: list[tuple[str, float]] = [
    # Strong negative
    ("misses",           -0.9), ("miss",              -0.9),
    ("below expectations", -0.90), ("below estimates", -0.90),
    ("worse than expected", -0.90),
    ("lowers guidance",  -0.95), ("cuts guidance",    -0.95),
    ("lowers outlook",   -0.90), ("cuts outlook",     -0.90),
    ("lowers forecast",  -0.85),
    ("disappoints",      -0.85), ("disappointing",    -0.85),
    ("warns",            -0.80), ("warning",          -0.75),
    ("loss",             -0.70), ("net loss",         -0.80),
    ("decline",          -0.65), ("declines",         -0.65),
    ("weak",             -0.70), ("weaker",           -0.70),
    ("slows",            -0.65), ("slowdown",         -0.65),
    ("fda rejects",      -0.95), ("complete response letter", -0.90),
    ("trial fails",      -0.95), ("failed trial",     -0.95),
    ("negative results", -0.90), ("no clinical benefit", -0.90),
    ("downgrade",        -0.75), ("downgraded",       -0.75),
    ("underperform",     -0.70), ("sell rating",      -0.70),
    ("lawsuit",          -0.60), ("investigation",    -0.60),
    ("subpoena",         -0.70), ("class action",     -0.70),
    ("recalls",          -0.75), ("recall",           -0.70),
    ("layoffs",          -0.65), ("restructuring",    -0.50),
    ("bankruptcy",       -0.90), ("default",          -0.85),
    ("cuts dividend",    -0.75), ("suspends dividend", -0.80),
]

# Negation words that flip the signal direction
_NEGATIONS = frozenset([
    "not", "no", "never", "didn't", "did not", "doesn't", "does not",
    "won't", "will not", "isn't", "is not", "wasn't", "was not",
    "fails to", "fail to", "unable to", "failed to",
])

# Magnitude amplifiers
_AMPLIFIERS = frozenset([
    "significantly", "materially", "substantially", "greatly",
    "sharply", "dramatically", "massively", "strongly",
])

_DAMPENERS = frozenset([
    "slightly", "modestly", "marginally", "somewhat", "partially",
])


# ---------------------------------------------------------------------------
# Sentiment scoring — rule-based
# ---------------------------------------------------------------------------

def _score_sentiment_rules(text: str) -> tuple[float, float]:
    """
    Rule-based sentiment scorer.

    Returns:
        (sentiment_score, confidence)
        sentiment_score: -1.0 (strongly bearish) → +1.0 (strongly bullish)
        confidence: 0.0 → 1.0 (how certain are we of the direction)
    """
    text_lower = text.lower()
    words = re.split(r"\W+", text_lower)
    word_set = set(words)

    # Amplification multiplier
    amp = 1.0
    for amplifier in _AMPLIFIERS:
        if amplifier in text_lower:
            amp = 1.25
            break
    for dampener in _DAMPENERS:
        if dampener in text_lower:
            amp = 0.75
            break

    raw_scores: list[float] = []

    def _is_negated(phrase: str) -> bool:
        """Check if phrase is preceded by a negation within 3 words."""
        idx = text_lower.find(phrase)
        if idx < 0:
            return False
        prefix = text_lower[max(0, idx - 40):idx]
        prefix_words = re.split(r"\W+", prefix)
        last_words = prefix_words[-4:]
        return any(neg in " ".join(last_words) for neg in _NEGATIONS)

    for phrase, score in _POSITIVE_SIGNALS:
        if phrase in text_lower:
            actual_score = (-score * amp) if _is_negated(phrase) else (score * amp)
            raw_scores.append(actual_score)

    for phrase, score in _NEGATIVE_SIGNALS:
        if phrase in text_lower:
            actual_score = (-score * amp) if _is_negated(phrase) else (score * amp)
            raw_scores.append(actual_score)

    if not raw_scores:
        return 0.0, 0.0

    # Aggregate: mean of signals, bounded to [-1, 1]
    agg = sum(raw_scores) / len(raw_scores)
    agg = max(-1.0, min(1.0, agg))

    # Confidence: higher when signals agree (low variance) and more signals present
    import statistics
    if len(raw_scores) > 1:
        try:
            stdev = statistics.stdev(raw_scores)
        except statistics.StatisticsError:
            stdev = 0.5
        agreement = max(0.0, 1.0 - stdev)
    else:
        agreement = 0.7  # Single signal — moderate confidence

    volume_boost = min(0.2, len(raw_scores) * 0.04)
    confidence = min(1.0, agreement + volume_boost)

    return round(agg, 4), round(confidence, 4)


# ---------------------------------------------------------------------------
# OpenAI sentiment (optional — used when API key is available)
# ---------------------------------------------------------------------------

async def _score_sentiment_openai(text: str) -> tuple[float, float] | None:
    """
    Use gpt-3.5-turbo to classify headline sentiment.
    Returns (sentiment_score, confidence) or None if unavailable.
    Cost: ~$0.001 per call — negligible at scan volumes.

    Structured prompt ensures reproducible JSON output.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import aiohttp
        prompt = (
            "You are a financial news sentiment classifier for a trading system. "
            "Given a news headline or summary about a stock, classify the sentiment "
            "for the stock price.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            "{\"sentiment\": \"POSITIVE\" or \"NEGATIVE\" or \"NEUTRAL\", "
            "\"score\": float from -1.0 to 1.0, "
            "\"confidence\": float from 0.0 to 1.0}\n\n"
            "Rules:\n"
            "- POSITIVE: news likely causes stock to go up (earnings beat, drug approval, acquisition premium)\n"
            "- NEGATIVE: news likely causes stock to go down (earnings miss, trial failure, guidance cut)\n"
            "- NEUTRAL: ambiguous, mixed, or macro news with unclear directional impact\n"
            "- score: -1.0 = strongly bearish, 0.0 = neutral, +1.0 = strongly bullish\n"
            "- confidence: how certain are you of the direction\n\n"
            f"Headline: {text[:500]}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80,
                    "temperature": 0.0,  # Deterministic
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()

                import json
                # Strip any markdown fences
                content = content.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(content)

                score      = float(parsed.get("score", 0.0))
                confidence = float(parsed.get("confidence", 0.5))
                return max(-1.0, min(1.0, score)), max(0.0, min(1.0, confidence))

    except Exception as exc:
        logger.debug("OpenAI sentiment failed (falling back to rules): {}", exc)
        return None


# ---------------------------------------------------------------------------
# Unified sentiment scorer
# ---------------------------------------------------------------------------

async def score_sentiment(text: str) -> tuple[NewsSentiment, float, float]:
    """
    Score the sentiment of a news text.

    Returns:
        (sentiment_enum, sentiment_score, sentiment_confidence)

    Tries OpenAI first (if API key set), falls back to rule-based.
    """
    # Try OpenAI
    openai_result = await _score_sentiment_openai(text)
    if openai_result is not None:
        score, confidence = openai_result
        logger.debug("OpenAI sentiment: score={:.2f} conf={:.2f}", score, confidence)
    else:
        score, confidence = _score_sentiment_rules(text)
        logger.debug("Rule-based sentiment: score={:.2f} conf={:.2f}", score, confidence)

    if abs(score) < 0.15 or confidence < 0.3:
        sentiment = NewsSentiment.NEUTRAL
    elif score > 0:
        sentiment = NewsSentiment.POSITIVE
    else:
        sentiment = NewsSentiment.NEGATIVE

    return sentiment, score, confidence


# ---------------------------------------------------------------------------
# Category classifier (unchanged logic, same as v1)
# ---------------------------------------------------------------------------

def classify_headline(text: str) -> ClassificationResult:
    """
    Classify category only (synchronous — no sentiment yet).
    Used when async context is unavailable.
    """
    text_lower = text.lower()
    text_lower = re.sub(r"[^\w\s]", " ", text_lower)

    best_category   = CatalystCategory.UNKNOWN
    best_confidence = 0.0
    best_keywords: list[str] = []

    for category, keywords, base_conf in _CATEGORY_KEYWORDS:
        matched = [kw for kw in keywords if kw in text_lower]
        if not matched:
            continue
        match_ratio = len(matched) / len(keywords)
        confidence  = min(1.0, base_conf * (1.0 + match_ratio * 0.2))
        if confidence > best_confidence:
            best_confidence = confidence
            best_category   = category
            best_keywords   = matched

    return ClassificationResult(
        category=best_category,
        confidence=best_confidence,
        matched_keywords=best_keywords,
    )


async def classify_with_sentiment(text: str) -> ClassificationResult:
    """
    Full classification: category + sentiment (async — may call OpenAI).
    This is what the catalyst engine should use.
    """
    # Stage 1: category
    result = classify_headline(text)

    # Stage 2: sentiment
    sentiment, score, sent_conf = await score_sentiment(text)
    result.sentiment            = sentiment
    result.sentiment_score      = score
    result.sentiment_confidence = sent_conf

    # Combined confidence: category_conf × sentiment_conf
    # If sentiment is NEGATIVE for a positive-category event, confidence drops.
    # If they agree (positive category + positive sentiment), confidence rises.
    if result.confidence > 0:
        category_direction = _infer_category_direction(result.category)
        direction_agreement = (
            (score > 0 and category_direction > 0) or
            (score < 0 and category_direction < 0) or
            category_direction == 0
        )
        agreement_mult = 1.1 if direction_agreement else 0.7
        result.combined_confidence = min(1.0,
            result.confidence * sent_conf * agreement_mult
        )
    else:
        result.combined_confidence = 0.0

    return result


def _infer_category_direction(category: CatalystCategory) -> float:
    """
    Expected sentiment direction for each category.
    +1 = typically positive, -1 = typically negative, 0 = neutral/mixed.
    """
    _DIRECTION: dict[CatalystCategory, float] = {
        CatalystCategory.EARNINGS:              0.0,   # Could be beat or miss
        CatalystCategory.EARNINGS_GUIDANCE:     0.0,   # Could be raised or cut
        CatalystCategory.FDA_OR_BIOTECH_NEWS:   0.0,   # Approval or rejection
        CatalystCategory.ANALYST_UPGRADE:       1.0,
        CatalystCategory.ANALYST_DOWNGRADE:    -1.0,
        CatalystCategory.M_AND_A:               1.0,   # Usually premium for target
        CatalystCategory.PARTNERSHIP_OR_CONTRACT: 0.5,
        CatalystCategory.PRODUCT_LAUNCH:        0.5,
        CatalystCategory.LEGAL_OR_REGULATORY:  -0.5,
        CatalystCategory.PRESS_RELEASE:         0.0,
        CatalystCategory.MACRO_RELATED:         0.0,
        CatalystCategory.UNKNOWN:               0.0,
    }
    return _DIRECTION.get(category, 0.0)


def classify_multiple(texts: list[str]) -> ClassificationResult:
    """Sync fallback: classify multiple texts, return highest-confidence result."""
    if not texts:
        return ClassificationResult(CatalystCategory.UNKNOWN, 0.0, [])
    results = [classify_headline(t) for t in texts if t]
    return max(results, key=lambda r: r.confidence)


async def classify_multiple_with_sentiment(texts: list[str]) -> ClassificationResult:
    """Async full classification across multiple text snippets."""
    if not texts:
        return ClassificationResult(CatalystCategory.UNKNOWN, 0.0, [])

    # Classify all with sentiment
    import asyncio
    results = await asyncio.gather(*[classify_with_sentiment(t) for t in texts if t])

    # Take highest combined_confidence result
    best = max(results, key=lambda r: r.combined_confidence or r.confidence)
    return best
