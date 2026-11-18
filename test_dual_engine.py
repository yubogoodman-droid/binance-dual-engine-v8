#!/usr/bin/env python3
"""離線單元測試：驗證結構雙引擎無幽靈保本、無前視流動性偏差、無字母排序偏誤與二拍搜尋邏輯。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import backtest_90d_breakout as bt
import binance_breakout_scanner as scanner

TPE = ZoneInfo("Asia/Taipei")


def _make_base_df(n: int = 320, base_px: float = 10.0, qv_per_bar: float = 100_000.0) -> pd.DataFrame:
    t0 = datetime(2026, 10, 1, 0, 0, tzinfo=TPE)
    idx = [t0 + timedelta(minutes=15 * k) for k in range(n)]
    closes = np.full(n, base_px, dtype=float)
    opens = closes * 0.999
    highs = closes * 1.002
    lows = closes * 0.998
    vols = np.full(n, qv_per_bar / base_px, dtype=float)
    qvs = np.full(n, qv_per_bar, dtype=float)
    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols, "QuoteVolume": qvs},
        index=pd.DatetimeIndex(idx),
    )
    return scanner.enrich_indicators(df, "15m")


class DualEngineAuditFixesTest(unittest.TestCase):
    def test_no_same_bar_phantom_breakeven_on_wick_reversal(self) -> None:
        """同根 K 線若僅上影線刺穿 +1.5R 但收盤砸回原點，絕不可啟動幽靈保本。"""
        df = _make_base_df(320, base_px=100.0)
        entry_idx = 250
        # 進場價 100，初始止損 96 (風險 4 = 4%)，+1.5R 目標 = 106
        sig = scanner.BreakoutSignal(
            symbol="TESTUSDT",
            interval="15m",
            signal_type="COILED_MA200_STAGE2",
            bar_idx=entry_idx,
            timestamp=df.index[entry_idx],
            entry_price=100.0,
            stop_price=96.0,
            target_2r=108.0,
            target_4r=116.0,
            vol_ratio=3.5,
            bar_gain_pct=1.5,
            ma_spread_pct=0.8,
            base_range_pct=1.2,
            clv=0.85,
            max_gain_after_pct=0.0,
            max_r_after=0.0,
            note="test",
            prev_24h_quote_vol=10_000_000.0,
        )
        # k = entry_idx + 1：上影線衝到 107 (>106)，但收盤砸回 99.0 (<106)
        df.iloc[entry_idx + 1, df.columns.get_loc("Open")] = 100.0
        df.iloc[entry_idx + 1, df.columns.get_loc("High")] = 107.0
        df.iloc[entry_idx + 1, df.columns.get_loc("Low")] = 98.5
        df.iloc[entry_idx + 1, df.columns.get_loc("Close")] = 99.0

        # k = entry_idx + 2：直接跳空開低於 95.0 (< 初始止損 96.0)，必須以開盤價 95.0 認列跳空止損！
        df.iloc[entry_idx + 2, df.columns.get_loc("Open")] = 95.0
        df.iloc[entry_idx + 2, df.columns.get_loc("High")] = 95.5
        df.iloc[entry_idx + 2, df.columns.get_loc("Low")] = 94.0
        df.iloc[entry_idx + 2, df.columns.get_loc("Close")] = 94.5

        tr = bt.simulate_structural_trade(df, sig)
        self.assertIsNotNone(tr)
        assert tr is not None
        self.assertEqual(tr.exit_reason, "初始止損")
        self.assertAlmostEqual(tr.exit_price, 95.0, places=6)
        self.assertLess(tr.pnl_pct_net, -5.0)

    def test_breakeven_stop_also_respects_open_gap_slippage(self) -> None:
        """即使前一根收盤正式站上 +1.5R 啟動保本，次根若跳空跌破保本價，也必須按實際開盤價成交。"""
        df = _make_base_df(320, base_px=100.0)
        entry_idx = 250
        sig = scanner.BreakoutSignal(
            symbol="TESTUSDT",
            interval="15m",
            signal_type="STAGE2_SHELF_BREAKOUT",
            bar_idx=entry_idx,
            timestamp=df.index[entry_idx],
            entry_price=100.0,
            stop_price=96.0,
            target_2r=108.0,
            target_4r=116.0,
            vol_ratio=3.0,
            bar_gain_pct=1.4,
            ma_spread_pct=1.0,
            base_range_pct=2.0,
            clv=0.80,
            max_gain_after_pct=0.0,
            max_r_after=0.0,
            note="test",
            prev_24h_quote_vol=10_000_000.0,
        )
        # k+1 收盤 106.5 >= 106.0，正式啟動保本 cur_stop = 100.3
        df.iloc[entry_idx + 1, df.columns.get_loc("Open")] = 100.0
        df.iloc[entry_idx + 1, df.columns.get_loc("High")] = 107.0
        df.iloc[entry_idx + 1, df.columns.get_loc("Low")] = 99.8
        df.iloc[entry_idx + 1, df.columns.get_loc("Close")] = 106.5

        # k+2 閃崩跳空開在 99.2 (< 保本價 100.3)，不可用 100.3 幽靈成交，必須成交在 99.2
        df.iloc[entry_idx + 2, df.columns.get_loc("Open")] = 99.2
        df.iloc[entry_idx + 2, df.columns.get_loc("High")] = 99.5
        df.iloc[entry_idx + 2, df.columns.get_loc("Low")] = 98.0
        df.iloc[entry_idx + 2, df.columns.get_loc("Close")] = 98.5

        tr = bt.simulate_structural_trade(df, sig)
        self.assertIsNotNone(tr)
        assert tr is not None
        self.assertEqual(tr.exit_reason, "保本鎖利止盈")
        self.assertAlmostEqual(tr.exit_price, 99.2, places=6)

    def test_point_in_time_24h_liquidity_rejects_post_pump_illiquid_coin(self) -> None:
        """進場前 24h 成交額不足 $5M 時必須攔截，即使訊號後暴漲爆量也不得事後入選。"""
        # 每根 15m 成交額僅 $10,000 → 24h (96根) = $0.96M < $5M
        n = 320
        t0 = datetime(2026, 10, 1, 0, 0, tzinfo=TPE)
        idx = [t0 + timedelta(minutes=15 * k) for k in range(n)]
        closes = np.linspace(10.5, 10.0, n)
        # 前 290 根壓在緩降->走平的 MA200 下方，295 穿線，296 二拍確認，300 後暴量 1000 倍
        closes[:200] = 10.20
        closes[200:295] = 9.95
        closes[295] = 10.12
        closes[296] = 10.28
        closes[297:] = 15.00
        opens = closes.copy()
        opens[295] = 9.95
        opens[296] = 10.12
        highs = np.maximum(opens, closes) * 1.001
        lows = np.minimum(opens, closes) * 0.998
        vols = np.full(n, 1000.0)
        vols[295] = 4000.0
        vols[296] = 5000.0
        vols[298:] = 5_000_000.0  # 事後才爆出幾千萬美金成交額
        qvs = closes * vols

        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols, "QuoteVolume": qvs},
            index=pd.DatetimeIndex(idx),
        )
        df = scanner.enrich_indicators(df, "15m")

        sigs_strict = scanner.detect_coiled_ma200_signals(df, "ILLIQUSDT", "15m", min_quote_vol_24h=5_000_000.0)
        sigs_loose = scanner.detect_coiled_ma200_signals(df, "ILLIQUSDT", "15m", min_quote_vol_24h=0.0)
        self.assertEqual(len(sigs_strict), 0)
        self.assertEqual(len(sigs_loose), 1)

    def test_ranking_eliminates_alphabetical_bias_and_filters_spike_clusters(self) -> None:
        """同時間訊號應依引擎與均線緊密度排序（而非字母 A 優先於 Z），且 >=3 檔同根齊噴時自動過濾。"""
        ts1 = datetime(2026, 10, 10, 12, 0, tzinfo=TPE)
        s_alpha = scanner.BreakoutSignal(
            "AAAUSDT", "15m", "STAGE2_SHELF_BREAKOUT", 250, ts1, 10, 9.5, 11, 12, 3.0, 1.5, 1.8, 3.0, 0.8, 0, 0, "", 8e6
        )
        s_zulu = scanner.BreakoutSignal(
            "ZZZUSDT", "15m", "STAGE2_SHELF_BREAKOUT", 250, ts1, 10, 9.5, 11, 12, 3.0, 1.5, 0.6, 3.0, 0.8, 0, 0, "", 12e6
        )
        ranked = scanner.filter_and_rank_signals([s_alpha, s_zulu], max_same_bar_signals=2)
        self.assertEqual([x.symbol for x in ranked], ["ZZZUSDT", "AAAUSDT"])

        # 同根出現 3 檔訊號視為大盤消息面插針，整批過濾
        s_mid = scanner.BreakoutSignal(
            "MMMUSDT", "15m", "STAGE2_SHELF_BREAKOUT", 250, ts1, 10, 9.5, 11, 12, 3.0, 1.5, 1.1, 3.0, 0.8, 0, 0, "", 9e6
        )
        clustered = scanner.filter_and_rank_signals([s_alpha, s_zulu, s_mid], max_same_bar_signals=2)
        self.assertEqual(len(clustered), 0)

    def test_beat2_climax_volume_is_rejected(self) -> None:
        """第二拍若爆出 >10x 天量（如 16x~24x 趕頂針），必須攔截不追高。"""
        n = 320
        t0 = datetime(2026, 10, 1, 0, 0, tzinfo=TPE)
        idx = [t0 + timedelta(minutes=15 * k) for k in range(n)]
        closes = np.full(n, 10.20)
        closes[200:295] = 9.95
        closes[295] = 10.12
        closes[296] = 10.28
        opens = closes.copy()
        opens[295] = 9.95
        opens[296] = 10.12
        highs = np.maximum(opens, closes) * 1.001
        lows = np.minimum(opens, closes) * 0.998
        vols = np.full(n, 20_000.0)
        vols[295] = 80_000.0    # 第一拍 4x
        vols[296] = 400_000.0   # 第二拍約 18x (>10x 天量衰竭)
        qvs = closes * vols

        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols, "QuoteVolume": qvs},
            index=pd.DatetimeIndex(idx),
        )
        df = scanner.enrich_indicators(df, "15m")
        sigs_capped = scanner.detect_coiled_ma200_signals(df, "CLIMAXUSDT", "15m")
        sigs_uncapped = scanner.detect_coiled_ma200_signals(df, "CLIMAXUSDT", "15m", follow_max_vol_ratio=99.0)
        self.assertEqual(len(sigs_capped), 0)
        self.assertEqual(len(sigs_uncapped), 1)


if __name__ == "__main__":
    unittest.main()
