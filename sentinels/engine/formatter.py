"""
Engine Prompt Formatter
Converts AggregatedMetrics into the engine prompt format
"""

from typing import Optional, Tuple
from sentinels.data.metrics import AggregatedMetrics

import logging

logger = logging.getLogger(__name__)


class EnginePromptFormatter:
    """Format sentiment metrics for Engine AI consumption."""

    def __init__(
        self,
        strong_threshold: float = 0.2,
        low_dispersion_threshold: float = 0.3,
        thresholds: Optional[dict] = None,
    ):
        """
        Initialize formatter.

        Args:
            strong_threshold: Threshold for strong sentiment (abs value)
            low_dispersion_threshold: Threshold for low disagreement
        """
        self.strong_threshold = strong_threshold
        self.low_dispersion_threshold = low_dispersion_threshold

        # Centralized threshold configuration (overridable)
        self.thresholds = thresholds or {
            # Dispersion thresholds
            "dispersion_low": 0.3,
            "dispersion_high": 0.7,
            # Impact thresholds
            "impact_strong_positive": 1.0,
            "impact_strong_negative": -1.0,
            # Bull/Bear ratio thresholds
            "bb_ratio_bullish": 1.5,
            "bb_ratio_bearish": 0.5,
            # Extreme ratio thresholds
            "extreme_moderate": 0.3,
            "extreme_overheated": 0.75,
            "extreme_mild": 0.5,
        }

    def format_for_engine(self, metrics: AggregatedMetrics, date: str) -> str:
        """
        Format aggregated metrics as Engine CustomPrompt with detailed interpretation.

        Args:
            metrics: Aggregated sentiment metrics
            date: Date string (YYYY-MM-DD)

        Returns:
            Formatted prompt text with comprehensive analysis
        """
        sections = []
        th = self.thresholds

        # ========== Header ==========
        sections.append(f"## News Sentiment Analysis Report ({date})")
        sections.append("")
        sections.append(f"**Data Source**: {metrics.news_count} articles analyzed from market news")
        sections.append("")

        # ========== Part 1: Sentiment distribution ==========
        sections.append("###  Sentiment Distribution")
        sections.append("")

        pos_pct = (
            (metrics.positive_count / metrics.news_count * 100) if metrics.news_count > 0 else 0
        )
        neg_pct = (
            (metrics.negative_count / metrics.news_count * 100) if metrics.news_count > 0 else 0
        )
        neu_pct = (
            (metrics.neutral_count / metrics.news_count * 100) if metrics.news_count > 0 else 0
        )

        sections.append(
            f"- **Positive**: {pos_pct:.1f}% ({metrics.positive_count} articles) | Avg Probability: {metrics.avg_positive:.3f}"
        )
        sections.append(
            f"- **Negative**: {neg_pct:.1f}% ({metrics.negative_count} articles) | Avg Probability: {metrics.avg_negative:.3f}"
        )
        sections.append(
            f"- **Neutral**: {neu_pct:.1f}% ({metrics.neutral_count} articles) | Avg Probability: {metrics.avg_neutral:.3f}"
        )
        sections.append("")

        # Overall sentiment tendency
        if pos_pct > neg_pct + 10:
            overall_sentiment = f" **BULLISH** (Positive dominates by {pos_pct - neg_pct:.1f}%)"
        elif neg_pct > pos_pct + 10:
            overall_sentiment = f" **BEARISH** (Negative dominates by {neg_pct - pos_pct:.1f}%)"
        else:
            overall_sentiment = (
                f"**NEUTRAL** (Balanced sentiment, diff: {abs(pos_pct - neg_pct):.1f}%)"
            )

        sections.append(f"**Overall Market Tone**: {overall_sentiment}")
        sections.append("")

        # ========== Part 2: Four core indicators + interpretation ==========
        sections.append("###  Core Sentiment Indicators (With Interpretation)")
        sections.append("")

        # Indicator A: Dispersion
        sections.append(f"**A. Market Disagreement (Dispersion): {metrics.dispersion:.3f}**")
        sections.append("")
        if metrics.dispersion < th["dispersion_low"]:
            sections.append(f"    **LOW Disagreement** (< {th['dispersion_low']})")
            sections.append("   → Market participants have strong consensus")
            sections.append("   → High confidence for trend-following strategies")
            sections.append("   → When combined with directional signals, conviction is high")
        elif metrics.dispersion > th["dispersion_high"]:
            sections.append(f"    **HIGH Disagreement** (> {th['dispersion_high']})")
            sections.append("   → Market participants strongly divided on direction")
            sections.append("   →  High uncertainty - reduce position sizes")
            sections.append("   → Expect choppy price action and potential volatility")
        else:
            sections.append(
                f"    **MODERATE Disagreement** ({th['dispersion_low']} - {th['dispersion_high']})"
            )
            sections.append("   → Normal market condition with mixed views")
            sections.append("   → Combine sentiment with technical signals for confirmation")
        sections.append("")

        # Indicator B: Impact
        sections.append(f"**B. Sentiment Impact: {metrics.impact:+.3f}**")
        sections.append("")
        if metrics.impact > th["impact_strong_positive"]:
            sections.append(f"   **STRONG Positive Sentiment** (> {th['impact_strong_positive']})")
            sections.append("   → Market enthusiasm is high with significant information content")
            sections.append("   → Consider LONG positions when technical indicators confirm")
            sections.append("   → Good environment for breakout trades")
        elif metrics.impact < th["impact_strong_negative"]:
            sections.append(f"    **PANIC Sentiment** (< {th['impact_strong_negative']})")
            sections.append("   → Fear and negative sentiment dominate the market")
            sections.append("   → Avoid new LONG positions")
            sections.append("   → Consider SHORT opportunities or stay in cash")
        elif metrics.impact > 0:
            sections.append(
                f"    **Mild Positive Sentiment** (0 to {th['impact_strong_positive']})"
            )
            sections.append("   → Slightly bullish bias but not strong conviction")
            sections.append("   → Use as confirming signal, not primary driver")
        elif metrics.impact < 0:
            sections.append(
                f"    **Mild Negative Sentiment** ({th['impact_strong_negative']} to 0)"
            )
            sections.append("   → Slightly bearish bias but not panic")
            sections.append("   → Be selective with LONG entries")
        else:
            sections.append("   **Neutral Sentiment** (~ 0)")
            sections.append("   → No clear directional bias from sentiment")
        sections.append("")

        # Indicator C: Bull/Bear ratio
        sections.append(f"**C. Bull/Bear Ratio: {metrics.bull_bear_ratio:.2f}**")
        sections.append("")
        if metrics.bull_bear_ratio > th["bb_ratio_bullish"]:
            sections.append(f"   **MAJORITY BULLISH** (> {th['bb_ratio_bullish']})")
            sections.append("   → Bulls significantly outnumber bears")
            sections.append("   → Strong buying sentiment in the market")
            sections.append("   → Favor LONG positions when setup quality is high")
        elif metrics.bull_bear_ratio < th["bb_ratio_bearish"]:
            sections.append(f"   **MAJORITY BEARISH** (< {th['bb_ratio_bearish']})")
            sections.append("   → Bears significantly outnumber bulls")
            sections.append("   → Strong selling sentiment in the market")
            sections.append("   → Avoid LONG positions, consider SHORT opportunities")
        else:
            sections.append(
                f"   **BALANCED** ({th['bb_ratio_bearish']} - {th['bb_ratio_bullish']})"
            )
            sections.append("   → Bulls and bears roughly equal in strength")
            sections.append("   → No clear directional bias from bull/bear ratio")
        sections.append("")

        # Indicator D: Extreme sentiment ratio
        sections.append(f"**D. Extreme Sentiment Ratio: {metrics.extreme_ratio:.1%}**")
        sections.append("")
        if metrics.extreme_ratio < th["extreme_moderate"]:
            sections.append(f"   **MODERATE Views** (< {th['extreme_moderate']:.0%})")
            sections.append("   → Market opinions are balanced and rational")
            sections.append("   → Healthy sentiment environment")
            sections.append("   → Normal trading conditions, no overextension")
        elif metrics.extreme_ratio > th["extreme_overheated"]:
            sections.append(f"   **OVERHEATED Market** (> {th['extreme_overheated']:.0%})")
            sections.append("   → Extreme sentiment dominates (either greed or fear)")
            sections.append("   → WARNING: Potential reversal risk")
            sections.append("   → Market may be overbought (bulls) or oversold (bears)")
            sections.append("   → Consider taking profits or waiting for cooldown")
        else:
            sections.append(
                f"   **MIXED Conviction** ({th['extreme_moderate']:.0%} - {th['extreme_overheated']:.0%})"
            )
            sections.append("   → Some strong opinions but not excessive")
            sections.append("   → Market has conviction but not at extreme levels")
        sections.append("")

        # ========== Part 3: Integrated trading recommendation ==========
        sections.append("###  Integrated Trading Recommendation")
        sections.append("")

        guidance_text, scenario = self._generate_integrated_guidance(metrics)
        sections.append(guidance_text)

        # Emit log output before returning
        prompt_text = "\n".join(sections)

        # Emit a concise summary
        logger.info(f"Sentiment Prompt Generated for {date}")
        logger.info(
            f"   Articles: {metrics.news_count} | "
            f"Mean: {metrics.sentiment_mean:+.4f} | "
            f"Impact: {metrics.impact:+.3f} | "
            f"BB Ratio: {metrics.bull_bear_ratio:.2f} | "
            f"Scenario: {scenario} | "
            f"Extreme Ratio: {metrics.extreme_ratio:.1%}"
        )

        # Full prompt output (debug mode)
        logger.debug(f"Full Prompt:\n{prompt_text}")

        return prompt_text

    def _generate_integrated_guidance(self, metrics: AggregatedMetrics) -> Tuple[str, str]:
        """
        Generate comprehensive trading guidance based on all four metrics.

        Args:
            metrics: Aggregated sentiment metrics

        Returns:
            Tuple of (guidance_text, scenario_type)
        """
        lines = []
        scenario = "NEUTRAL"  # default scenario

        th = self.thresholds

        # Extract the four indicators
        dispersion = metrics.dispersion
        impact = metrics.impact
        bb_ratio = metrics.bull_bear_ratio
        extreme_ratio = metrics.extreme_ratio

        # Scenario 1: Strong bullish setup (low dispersion + strong impact + bull-dominated + mild extremes)
        if (
            dispersion < th["dispersion_low"]
            and impact > th["impact_strong_positive"]
            and bb_ratio > th["bb_ratio_bullish"]
            and extreme_ratio < th["extreme_mild"]
        ):
            scenario = "STRONG_BULLISH"
            lines.append(" **STRONG BULLISH SETUP** (All Green Lights)")
            lines.append("")
            lines.append("**Why**: All four indicators align for a bullish scenario:")
            lines.append("- Low disagreement → Strong consensus")
            lines.append("- High positive impact → Market enthusiasm")
            lines.append("- Bulls dominate bears → Buying pressure")
            lines.append("- Sentiment not overheated → Room to run")
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("- **Direction**: Favor LONG positions")
            lines.append(
                "- **Entry Timing**: When technicals confirm (EMA cross, breakout, volume surge)"
            )
            lines.append("- **Position Size**: Normal to slightly aggressive (within risk limits)")
            lines.append("- **Risk Management**: Set stop-loss but let profits run")
            lines.append("")
            lines.append(
                f"**Watch For**: Extreme ratio climbing above {th['extreme_overheated']:.0%} (overheating warning)"
            )

        # Scenario 2: Strong bearish setup (low dispersion + panic impact + bear-dominated)
        elif (
            dispersion < th["dispersion_low"]
            and impact < th["impact_strong_negative"]
            and bb_ratio < th["bb_ratio_bearish"]
        ):
            scenario = "STRONG_BEARISH"
            lines.append(" **STRONG BEARISH SETUP** (Red Alert)")
            lines.append("")
            lines.append("**Why**: All indicators align for a bearish scenario:")
            lines.append("- Low disagreement → Panic consensus")
            lines.append("- Negative impact → Market fear dominant")
            lines.append("- Bears dominate bulls → Selling pressure")
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("- **Direction**: AVOID new LONG positions")
            lines.append("- **Opportunities**: Consider SHORT on technical breakdown")
            lines.append("- **Position Size**: Reduce overall exposure")
            lines.append("- **Risk Management**: Prioritize capital preservation")
            lines.append("")
            lines.append("**Watch For**: Sentiment stabilization or capitulation reversal signals")

        # Scenario 3: Overheating warning (extreme_ratio > 0.7)
        elif extreme_ratio > th["extreme_overheated"]:
            scenario = "OVERHEATED_BULLISH" if impact > 0 else "OVERHEATED_BEARISH"
            lines.append(" **MARKET OVERHEATING WARNING**")
            lines.append("")
            lines.append(
                f"**Alert**: Extreme sentiment ratio at {extreme_ratio:.1%} (>{th['extreme_overheated']:.0%} threshold)"
            )
            lines.append("")
            if impact > 0:
                lines.append("**Situation**: Too many bulls with extreme conviction")
                lines.append("- Market may be overbought")
                lines.append("- Excessive greed/FOMO detected")
                lines.append("")
                lines.append("**Trading Strategy**:")
                lines.append("-  Be **VERY cautious** with new LONG entries")
                lines.append("- Consider **taking profits** on existing positions")
                lines.append("- Watch for **reversal signals** (volume decline, RSI divergence)")
                lines.append("")
                lines.append("**Typical Pattern**: Sentiment extremes often precede corrections")
            else:
                lines.append("**Situation**: Extreme fear/panic in the market")
                lines.append("- Market may be oversold")
                lines.append("- Excessive fear detected")
                lines.append("")
                lines.append("**Trading Strategy**:")
                lines.append("-  Potential **contrarian opportunity** emerging")
                lines.append("-  But **wait for stabilization** first")
                lines.append(
                    "-  Look for **capitulation signals** (volume spike + price stabilization)"
                )
                lines.append("")
                lines.append("**Warning**: Don't catch a falling knife - let sentiment settle")

        # Scenario 4: High-disagreement market (dispersion > 0.7)
        elif dispersion > th["dispersion_high"]:
            scenario = "HIGH_DISAGREEMENT"
            lines.append(" **HIGH DISAGREEMENT MARKET**")
            lines.append("")
            lines.append(
                f"**Alert**: Market dispersion at {dispersion:.3f} (>{th['dispersion_high']} threshold)"
            )
            lines.append("")
            lines.append("**Situation**: No clear consensus - bulls and bears equally vocal")
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("-  **Reduce position sizes** by 30-50%")
            lines.append("-  **Tighten stop-losses** (expect volatility)")
            lines.append("-  **Avoid** opening new positions unless conviction is extreme")
            lines.append(
                f"-  **Wait** for sentiment to consolidate (dispersion drops below {th['dispersion_low']})"
            )
            lines.append("")
            lines.append("**Expected Market Behavior**: Choppy, range-bound, unpredictable swings")

        # Scenario 5: Moderate bullish (mid-range indicators, leaning long)
        elif impact > 0 and bb_ratio > 1.0:
            scenario = "MODERATE_BULLISH"
            lines.append(" **MODERATE BULLISH SENTIMENT**")
            lines.append("")
            lines.append("**Market Conditions**:")
            lines.append(f"- Impact: {impact:.3f} (positive but < {th['impact_strong_positive']})")
            lines.append(f"- Bull/Bear: {bb_ratio:.2f} (bulls ahead)")
            lines.append(
                f"- Dispersion: {dispersion:.3f} ({'low' if dispersion < th['dispersion_low'] else 'moderate'} disagreement)"
            )
            lines.append(
                f"- Extreme Ratio: {extreme_ratio:.1%} ({'moderate' if extreme_ratio < th['extreme_mild'] else 'elevated'})"
            )
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("-  Mild bullish bias - use sentiment as **confirming signal**")
            lines.append("-  Enter LONG when technical setups align (golden cross, breakout)")
            lines.append("-  Normal position sizing - don't overtrade on sentiment alone")
            lines.append("-  Maintain balanced risk management")

        # Scenario 6: Moderate bearish (mid-range indicators, leaning short)
        elif impact < 0 and bb_ratio < 1.0:
            scenario = "MODERATE_BEARISH"
            lines.append(" **MODERATE BEARISH SENTIMENT**")
            lines.append("")
            lines.append("**Market Conditions**:")
            lines.append(f"- Impact: {impact:.3f} (negative but > {th['impact_strong_negative']})")
            lines.append(f"- Bull/Bear: {bb_ratio:.2f} (bears ahead)")
            lines.append(
                f"- Dispersion: {dispersion:.3f} ({'low' if dispersion < th['dispersion_low'] else 'moderate'} disagreement)"
            )
            lines.append(f"- Extreme Ratio: {extreme_ratio:.1%}")
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("-  Mild bearish bias - be **selective** with LONG entries")
            lines.append("-  Only take LONG on **high-quality setups** with strong technicals")
            lines.append("-  Consider **smaller position sizes** (defensive)")
            lines.append("-  Prioritize risk management over aggressive gains")

        # Scenario 7: Neutral market (no clear direction)
        else:
            lines.append(" **NEUTRAL MARKET SENTIMENT**")
            lines.append("")
            lines.append("**Market Conditions**:")
            lines.append(f"- Impact: {impact:.3f} (near zero)")
            lines.append(f"- Bull/Bear: {bb_ratio:.2f} (balanced)")
            lines.append(f"- Dispersion: {dispersion:.3f}")
            lines.append(f"- Extreme Ratio: {extreme_ratio:.1%}")
            lines.append("")
            lines.append("**Interpretation**: Sentiment indicators show no clear directional bias")
            lines.append("")
            lines.append("**Trading Strategy**:")
            lines.append("-  Rely primarily on **technical analysis** and price action")
            lines.append("-  Focus on **chart patterns**, support/resistance, volume")
            lines.append("-  Normal risk management protocols")
            lines.append("-  Wait for stronger sentiment signals or clear technical triggers")

        lines.append("")
        return "\n".join(lines), scenario
