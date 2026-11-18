#!/usr/bin/env python3
"""幣安永續合約「起漲點策略 v2」90天完整實盤級回測 + HTML 視覺化戰績看板生成器

用法：
  python backtest_90d_breakout.py --interval 15m --days 90 --top 40 --html
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import backtest_utils as bt_utils
import binance_breakout_scanner as scanner

logger = logging.getLogger("backtest_90d_breakout")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

TPE = ZoneInfo("Asia/Taipei")
FEE_ROUNDTRIP_PCT = 0.12  # 0.12% 來回手續費+滑價


def fetch_klines_paginated(
    symbol: str,
    interval: str = "15m",
    days: int = 90,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """分頁拉取過去指定天數的完整已收盤 K 線資料（支援選擇性本地快取）。"""
    bar_ms = scanner.INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = days * bars_per_day + 250

    if cache_dir is not None:
        cfile = cache_dir / f"{symbol}_{interval}_{days}d.pkl"
        if cfile.is_file():
            try:
                cached = pd.read_pickle(cfile)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    return cached
            except Exception as exc:
                logger.warning("讀取快取失敗 (%s): %s", cfile, exc)

    raw: list = []
    end_ms: Optional[int] = None
    now_ms = int(time.time() * 1000)

    while len(raw) < need:
        limit = min(1000, need - len(raw))
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            chunk = scanner.get_json("/fapi/v1/klines", params=params)
        except Exception as exc:
            logger.warning("拉取 %s (%s) K 線中斷: %s", symbol, interval, exc)
            break
        if not chunk:
            break
        raw = list(chunk) + raw
        end_ms = int(chunk[0][0]) - 1
        if len(chunk) < limit:
            break
        time.sleep(0.04)

    if not raw:
        return pd.DataFrame()

    while raw and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
            "QuoteVolume": [float(x[7]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()

    for period, _, _, _ in scanner.MA_SPECS:
        df[f"MA{period}"] = df["Close"].rolling(period, min_periods=period).mean()
    df["MA240"] = df["Close"].rolling(240, min_periods=240).mean()

    df["VOL_MA5"] = df["Volume"].rolling(5, min_periods=3).mean()
    df["VOL_MA10"] = df["Volume"].rolling(10, min_periods=5).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["VOL_MA20_PREV"] = df["Volume"].shift(1).rolling(20, min_periods=10).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20_PREV"].replace(0, np.nan)

    if cache_dir is not None and not df.empty:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(cache_dir / f"{symbol}_{interval}_{days}d.pkl")
        except Exception as exc:
            logger.warning("寫入快取失敗 (%s): %s", symbol, exc)
    return df


@dataclass
class TradeResult:
    symbol: str
    interval: str
    signal_type: str
    entry_idx: int
    exit_idx: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    stop_price: float
    exit_price: float
    risk_pct: float
    vol_ratio: float
    ma_spread_pct: float
    base_range_pct: float
    pnl_pct_fixed3r: float
    r_fixed3r: float
    pnl_pct_ma25: float
    r_ma25: float
    pnl_pct_hybrid: float
    r_hybrid: float
    exit_reason_ma25: str
    hold_bars: int
    note: str
    chart_file: str = ""
    funding_cost_pct: float = 0.0
    funding_source: str = "est"
    worst_low_price: float = 0.0
    bar_quote_vol_usd: float = 0.0
    trailing_24h_qv_usd: float = 0.0
    is_oos: bool = False
    risk_executed: bool = False
    risk_acct_ret_pct: float = 0.0
    risk_after_usd: float = 0.0
    roll_executed: bool = False
    roll_ret_pct: float = 0.0
    roll_after_usd: float = 0.0


def simulate_trade(
    df: pd.DataFrame,
    sig: scanner.BreakoutSignal,
    max_hold_bars: int = 96,
    funding_store: Optional[bt_utils.FundingRateStore] = None,
    trailing_24h_qv_usd: float = 0.0,
) -> Optional[TradeResult]:
    """對單筆訊號模擬三種出場方式（無未來函數，保守同根K線上移止損後檢查低點，扣費滑與資金費率）。"""
    i = sig.bar_idx
    if i >= len(df) - 2:
        return None

    entry = float(sig.entry_price)
    stop = float(sig.stop_price)
    risk = entry - stop
    if entry <= 0 or risk <= 0:
        return None

    risk_pct = risk / entry * 100.0
    if risk_pct > 15.0 or risk_pct < 0.8:
        return None

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    ma25 = df["MA25"].to_numpy(float)
    qv_arr = bt_utils.extract_quote_volume_array(df)
    times = df.index
    bar_ms = scanner.INTERVAL_MS.get(sig.interval, 900_000)
    end_idx = min(len(df) - 1, i + max_hold_bars)

    # 1. Fixed 3R
    exit_3r = closes[end_idx]
    exit_3r_idx = end_idx
    for j in range(i + 1, end_idx + 1):
        if lows[j] <= stop:
            exit_3r = float(min(opens[j], stop))
            exit_3r_idx = j
            break
        if highs[j] >= entry + 3.0 * risk:
            exit_3r = entry + 3.0 * risk
            exit_3r_idx = j
            break
    fc_3r = 0.0
    if funding_store is not None:
        fc_3r, _, _ = funding_store.compute_funding_cost_pct(
            sig.symbol, bt_utils.ts_to_ms(times[i]), bt_utils.ts_to_ms(times[exit_3r_idx]), bar_ms=bar_ms
        )
    pnl_3r = (exit_3r - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT - fc_3r
    r_3r = pnl_3r / risk_pct

    # 2. 優化版 v3 MA25 移動停利 (漲破 +0.7R 啟動保本+手續費防衛，同根保守檢查低點，收盤跌破 MA25 出場)
    exit_ma25 = float(closes[end_idx])
    exit_idx_ma25 = end_idx
    reason_ma25 = "到期平倉"
    cur_stop = stop
    armed_be = False
    worst_low = float(entry)
    for j in range(i + 1, end_idx + 1):
        worst_low = min(worst_low, float(lows[j]))
        if lows[j] <= cur_stop:
            exit_ma25 = float(min(opens[j], cur_stop))
            exit_idx_ma25 = j
            reason_ma25 = "保本止盈" if armed_be else "初始止損"
            break
        if not armed_be and highs[j] >= entry + 0.7 * risk:
            armed_be = True
            cur_stop = max(cur_stop, entry * 1.003)
            if lows[j] <= cur_stop:
                exit_ma25 = float(cur_stop)
                exit_idx_ma25 = j
                reason_ma25 = "保本止盈"
                break
        # 動能時間止損：進場後第 3 根 K 線若未發動且收盤跌破進場價 -1.8%，立即提早砍倉減損
        if j == i + 3 and closes[j] < entry * 0.982:
            exit_ma25 = float(closes[j])
            exit_idx_ma25 = j
            reason_ma25 = "3K動能停滯減損"
            break
        if j - i >= 2 and np.isfinite(ma25[j]) and closes[j] < ma25[j]:
            exit_ma25 = float(closes[j])
            exit_idx_ma25 = j
            reason_ma25 = "破MA25止盈" if closes[j] > entry * 1.0012 else "破MA25微損出場"
            break

    funding_cost_pct = 0.0
    funding_source = "none"
    if funding_store is not None:
        funding_cost_pct, _, funding_source = funding_store.compute_funding_cost_pct(
            sig.symbol, bt_utils.ts_to_ms(times[i]), bt_utils.ts_to_ms(times[exit_idx_ma25]), bar_ms=bar_ms
        )
    pnl_ma25 = (exit_ma25 - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT - funding_cost_pct
    r_ma25 = pnl_ma25 / risk_pct

    # 3. Hybrid: 觸及 +2R 平半倉 + 移保本，剩下一半沿 MA25 收盤跌破出場
    cur_stop = stop
    took_half = False
    half1_pnl_pct = 0.0
    half2_exit = float(closes[end_idx])
    half2_idx = end_idx

    for j in range(i + 1, end_idx + 1):
        if lows[j] <= cur_stop:
            half2_exit = float(min(opens[j], cur_stop))
            half2_idx = j
            break
        if not took_half and highs[j] >= entry + 2.0 * risk:
            took_half = True
            fc_h1 = 0.0
            if funding_store is not None:
                fc_h1, _, _ = funding_store.compute_funding_cost_pct(
                    sig.symbol, bt_utils.ts_to_ms(times[i]), bt_utils.ts_to_ms(times[j]), bar_ms=bar_ms
                )
            half1_pnl_pct = (2.0 * risk) / entry * 100.0 - FEE_ROUNDTRIP_PCT - fc_h1
            cur_stop = max(cur_stop, entry * 1.002)
            if lows[j] <= cur_stop:
                half2_exit = float(cur_stop)
                half2_idx = j
                break
        if j - i >= 2 and np.isfinite(ma25[j]) and closes[j] < ma25[j]:
            half2_exit = float(closes[j])
            half2_idx = j
            break

    fc_h2 = 0.0
    if funding_store is not None:
        fc_h2, _, _ = funding_store.compute_funding_cost_pct(
            sig.symbol, bt_utils.ts_to_ms(times[i]), bt_utils.ts_to_ms(times[half2_idx]), bar_ms=bar_ms
        )
    half2_pnl_pct = (half2_exit - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT - fc_h2
    pnl_hybrid = 0.5 * half1_pnl_pct + 0.5 * half2_pnl_pct if took_half else half2_pnl_pct
    r_hybrid = pnl_hybrid / risk_pct

    return TradeResult(
        symbol=sig.symbol,
        interval=sig.interval,
        signal_type=sig.signal_type,
        entry_idx=i,
        exit_idx=exit_idx_ma25,
        entry_time=sig.timestamp,
        exit_time=times[exit_idx_ma25],
        entry_price=entry,
        stop_price=stop,
        exit_price=exit_ma25,
        risk_pct=risk_pct,
        vol_ratio=sig.vol_ratio,
        ma_spread_pct=sig.ma_spread_pct,
        base_range_pct=sig.base_range_pct,
        pnl_pct_fixed3r=pnl_3r,
        r_fixed3r=r_3r,
        pnl_pct_ma25=pnl_ma25,
        r_ma25=r_ma25,
        pnl_pct_hybrid=pnl_hybrid,
        r_hybrid=r_hybrid,
        exit_reason_ma25=reason_ma25,
        hold_bars=exit_idx_ma25 - i,
        note=sig.note,
        funding_cost_pct=funding_cost_pct,
        funding_source=funding_source,
        worst_low_price=worst_low,
        bar_quote_vol_usd=float(qv_arr[i]) if i < len(qv_arr) else 0.0,
        trailing_24h_qv_usd=float(trailing_24h_qv_usd),
    )


def simulate_coiled_trade(
    df: pd.DataFrame,
    sig: scanner.BreakoutSignal,
    max_hold_bars: int = 288,
    arm_r: float = 1.2,
    arm_abs_pct: float = 3.60,
    max_risk_pct: float = 7.20,
    funding_store: Optional[bt_utils.FundingRateStore] = None,
    trailing_24h_qv_usd: float = 0.0,
) -> Optional[TradeResult]:
    """引擎 B（牛來）專屬出場模擬：觸及 1.2R 或 +3.6% 啟動保本鎖利（同根保守檢查低點），沿 MA99 移動停利，扣費滑與資金費率。"""
    i = sig.bar_idx
    if i >= len(df) - 2:
        return None

    entry = float(sig.entry_price)
    stop = float(sig.stop_price)
    risk = entry - stop
    if entry <= 0 or risk <= 0:
        return None

    risk_pct = risk / entry * 100.0
    if risk_pct > max_risk_pct or risk_pct < 0.8:
        return None

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    ma99 = df["MA99"].to_numpy(float)
    qv_arr = bt_utils.extract_quote_volume_array(df)
    times = df.index
    bar_ms = scanner.INTERVAL_MS.get(sig.interval, 900_000)
    end_idx = min(len(df) - 1, i + max_hold_bars)

    cur_stop = stop
    armed = False
    exit_idx = end_idx
    exit_px = float(closes[end_idx])
    reason = "72h到期平倉"
    arm_target = min(entry + arm_r * risk, entry * (1.0 + arm_abs_pct / 100.0))
    worst_low = float(entry)

    for k in range(i + 1, end_idx + 1):
        worst_low = min(worst_low, float(lows[k]))
        if lows[k] <= cur_stop:
            exit_idx = k
            exit_px = float(min(opens[k], cur_stop))
            reason = "保本鎖利止盈" if (armed and cur_stop >= entry) else "初始止損"
            break
        if not armed and highs[k] >= arm_target:
            armed = True
            cur_stop = max(cur_stop, entry * 1.0025)
            if lows[k] <= cur_stop:
                exit_idx = k
                exit_px = float(cur_stop)
                reason = "保本鎖利止盈"
                break
        if armed and np.isfinite(ma99[k]) and closes[k] < ma99[k]:
            exit_idx = k
            exit_px = float(closes[k])
            reason = "破MA99(1H MA25)止盈"
            break

    funding_cost_pct = 0.0
    funding_source = "none"
    if funding_store is not None:
        funding_cost_pct, _, funding_source = funding_store.compute_funding_cost_pct(
            sig.symbol, bt_utils.ts_to_ms(times[i]), bt_utils.ts_to_ms(times[exit_idx]), bar_ms=bar_ms
        )

    raw_pnl = (exit_px / entry - 1.0) * 100.0
    pnl_net = raw_pnl - FEE_ROUNDTRIP_PCT - funding_cost_pct
    r_net = pnl_net / risk_pct

    return TradeResult(
        symbol=sig.symbol,
        interval=sig.interval,
        signal_type=sig.signal_type,
        entry_idx=i,
        exit_idx=exit_idx,
        entry_time=sig.timestamp,
        exit_time=times[exit_idx],
        entry_price=entry,
        stop_price=stop,
        exit_price=exit_px,
        risk_pct=risk_pct,
        vol_ratio=sig.vol_ratio,
        ma_spread_pct=sig.ma_spread_pct,
        base_range_pct=sig.base_range_pct,
        pnl_pct_fixed3r=pnl_net,
        r_fixed3r=r_net,
        pnl_pct_ma25=pnl_net,
        r_ma25=r_net,
        pnl_pct_hybrid=pnl_net,
        r_hybrid=r_net,
        exit_reason_ma25=reason,
        hold_bars=exit_idx - i,
        note=sig.note,
        funding_cost_pct=funding_cost_pct,
        funding_source=funding_source,
        worst_low_price=worst_low,
        bar_quote_vol_usd=float(qv_arr[i]) if i < len(qv_arr) else 0.0,
        trailing_24h_qv_usd=float(trailing_24h_qv_usd),
    )


def plot_single_trade_card(df: pd.DataFrame, tr: TradeResult, out_path: Path) -> None:
    """為單筆回測交易繪製包含進場、初始止損線、MA25 出場點的專屬復盤圖。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    is_coiled = tr.signal_type == "COILED_MA200_STAGE2"
    left = max(0, tr.entry_idx - (55 if is_coiled else 45))
    right = min(len(df) - 1, min(tr.entry_idx + 145, tr.exit_idx + 20) if is_coiled else tr.exit_idx + 25)
    sub = df.iloc[left : right + 1].copy()
    if sub.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.06},
        facecolor="#ffffff",
    )

    x = np.arange(len(sub))
    opens = sub["Open"].to_numpy(float)
    highs = sub["High"].to_numpy(float)
    lows = sub["Low"].to_numpy(float)
    closes = sub["Close"].to_numpy(float)
    vols = sub["Volume"].to_numpy(float)
    times = sub.index

    up_color = "#0ecb81"
    down_color = "#f6465d"

    for idx in range(len(sub)):
        is_up = closes[idx] >= opens[idx]
        c = up_color if is_up else down_color
        ax_price.plot([idx, idx], [lows[idx], highs[idx]], color=c, linewidth=1.0, zorder=2)
        body_low = min(opens[idx], closes[idx])
        body_h = max(abs(closes[idx] - opens[idx]), (highs[idx] - lows[idx]) * 0.02, 1e-8)
        ax_price.add_patch(Rectangle((idx - 0.32, body_low), 0.64, body_h, facecolor=c, edgecolor=c, zorder=3))
        ax_vol.bar(idx, vols[idx], width=0.68, color=c, alpha=0.85, zorder=2)

    for period, color, width, label in scanner.MA_SPECS:
        col = f"MA{period}"
        if col in sub.columns and sub[col].notna().any():
            ax_price.plot(x, sub[col].to_numpy(float), color=color, linewidth=width, label=label, zorder=4)

    if "VOL_MA5" in sub.columns:
        ax_vol.plot(x, sub["VOL_MA5"].to_numpy(float), color="#f0b90b", linewidth=1.1, label="VOL MA(5)")
    if "VOL_MA10" in sub.columns:
        ax_vol.plot(x, sub["VOL_MA10"].to_numpy(float), color="#7e57c2", linewidth=1.1, label="VOL MA(10)")

    e_pos = tr.entry_idx - left
    x_pos = min(len(sub) - 1, tr.exit_idx - left)

    # 修正成交量副圖 Y 軸比例：避免主升段後期的天量把起漲當下的爆量柱壓扁成一條線
    entry_vol = float(vols[e_pos]) if 0 <= e_pos < len(vols) else 0.0
    max_vol_all = float(np.nanmax(vols)) if len(vols) else 1.0
    if entry_vol > 0 and max_vol_all > entry_vol * 2.5:
        vol_ylim_top = entry_vol * 2.35
        ax_vol.set_ylim(0, vol_ylim_top)
        vol_tip_y = entry_vol
    else:
        vol_ylim_top = max_vol_all * 1.18
        ax_vol.set_ylim(0, vol_ylim_top)
        vol_tip_y = entry_vol

    entry_badge_color = "#6a1b9a" if is_coiled else "#d50000"
    vol_label_prefix = "★ 牛來穿MA200二拍" if is_coiled else "★ 起漲爆量"
    if entry_vol > 0:
        ax_vol.annotate(
            f"{vol_label_prefix} {tr.vol_ratio:.1f}x",
            xy=(e_pos, min(vol_tip_y, vol_ylim_top * 0.92)),
            xytext=(max(1, e_pos - 13), min(vol_ylim_top * 0.76, max(vol_tip_y * 1.15, vol_ylim_top * 0.45))),
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": entry_badge_color, "alpha": 0.90, "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": entry_badge_color, "lw": 1.2},
            zorder=9,
        )

    # 畫進場與止損水平虛線
    ax_price.hlines(
        tr.entry_price,
        e_pos,
        x_pos,
        colors="#1e88e5",
        linestyles="--",
        linewidth=1.3,
        label=f"進場價 ({tr.entry_price:.5g})",
        zorder=5,
    )
    ax_price.hlines(
        tr.stop_price,
        max(0, e_pos - 5),
        min(len(sub) - 1, e_pos + 18),
        colors="#e53935",
        linestyles=":",
        linewidth=1.4,
        label=f"初始止損 ({tr.stop_price:.5g})",
        zorder=5,
    )

    # 進場與出場箭頭
    ax_price.scatter([e_pos], [lows[e_pos] * 0.97], marker="^", s=180, color=entry_badge_color, edgecolors="white", zorder=8)
    exit_badge_color = "#00897b" if tr.r_ma25 > 0 else "#546e7a"
    ax_price.scatter(
        [x_pos], [highs[x_pos] * 1.03], marker="v", s=180, color=exit_badge_color, edgecolors="white", zorder=8
    )

    y_span = float(np.nanmax(highs) - np.nanmin(lows))
    ax_price.annotate(
        f"★ 進場 {tr.entry_price:.5g}\n({tr.entry_time.strftime('%m-%d %H:%M')}, 量 {tr.vol_ratio:.1f}x)",
        xy=(e_pos, lows[e_pos]),
        xytext=(max(1, e_pos - 14), max(float(np.nanmin(lows)), lows[e_pos] - 0.14 * y_span)),
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": entry_badge_color, "alpha": 0.92, "edgecolor": "none"},
        arrowprops={"arrowstyle": "->", "color": entry_badge_color, "lw": 1.4},
        zorder=9,
    )
    ax_price.annotate(
        f"▼ {tr.exit_reason_ma25} {tr.exit_price:.5g}\n淨利: {tr.pnl_pct_ma25:+.1f}% ({tr.r_ma25:+.2f}R)",
        xy=(x_pos, highs[x_pos]),
        xytext=(max(2, min(len(sub) - 22, x_pos - 10)), min(float(np.nanmax(highs)) * 0.96, highs[x_pos] + 0.08 * y_span)),
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.38", "facecolor": exit_badge_color, "alpha": 0.92, "edgecolor": "none"},
        arrowprops={"arrowstyle": "->", "color": exit_badge_color, "lw": 1.4},
        zorder=9,
    )

    if tr.signal_type == "BOTTOM_BREAKOUT":
        stype_label = "引擎A·底部爆量起漲"
    elif tr.signal_type == "COILED_MA200_STAGE2":
        stype_label = "引擎B·牛來黏帶穿MA200二拍"
    else:
        stype_label = "引擎A·回踩MA25二次起漲"
    ax_price.set_title(
        f"{tr.symbol} ({tr.interval}) [{stype_label}] — 實戰淨利 {tr.pnl_pct_ma25:+.1f}% ({tr.r_ma25:+.2f}R)",
        fontsize=12.5,
        fontweight="bold",
    )
    ax_price.grid(True, linestyle="--", alpha=0.25)
    ax_price.legend(loc="upper left", ncol=5, fontsize=8, framealpha=0.85)
    ax_vol.grid(True, linestyle="--", alpha=0.25)

    tick_step = max(1, len(sub) // 8)
    tick_indices = list(range(0, len(sub), tick_step))
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels([times[k].strftime("%m-%d\n%H:%M") for k in tick_indices], fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135, bbox_inches="tight")
    plt.close(fig)


def _to_portfolio_inputs(trades: Sequence[TradeResult]) -> List[bt_utils.PortfolioTradeInput]:
    return [
        bt_utils.PortfolioTradeInput(
            symbol=t.symbol,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            entry_price=t.entry_price,
            stop_price=t.stop_price,
            exit_price=t.exit_price,
            risk_pct=t.risk_pct,
            net_pnl_pct=t.pnl_pct_ma25,
            r_mult=t.r_ma25,
            worst_low_price=t.worst_low_price if t.worst_low_price > 0 else min(t.entry_price, t.exit_price),
            bar_quote_vol_usd=t.bar_quote_vol_usd,
            trailing_24h_qv_usd=t.trailing_24h_qv_usd,
            funding_cost_pct=t.funding_cost_pct,
            priority=float(t.vol_ratio),
            payload=t,
        )
        for t in trades
    ]


def compute_3x_rolling(
    trades: Sequence[TradeResult],
    init_cap: float = 1000.0,
    leverage: float = 3.0,
    risk_pct_per_trade: float = 2.0,
    max_positions: int = 3,
    apply_liquidity_cap: bool = True,
    max_bar_vol_share: float = 0.02,
    base_mmr: float = 0.015,
) -> Dict[str, Any]:
    """誠實資金模擬：2% 以損定倉併發複利(主指標) + 1x 真實複利 + 3x 含強平與流動性上限(理論對照)。"""
    inputs = _to_portfolio_inputs(trades)
    res = bt_utils.simulate_honest_portfolio(
        inputs,
        initial_cap=init_cap,
        risk_pct_per_trade=risk_pct_per_trade,
        max_positions=max_positions,
        leverage_3x=leverage,
        slots_3x=1,
        base_mmr=base_mmr,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
    )
    hr = res["headline_risk"]
    sp = res["spot_1x"]
    c3 = res["comp_3x"]
    f3 = res["fixed_3x"]

    risk_map = {(rec.trade.symbol, rec.trade.entry_time): rec for rec in hr["records"]}
    roll_map = {(rec.trade.symbol, rec.trade.entry_time): rec for rec in c3["records"]}
    last_risk_eq = init_cap
    last_roll_eq = init_cap
    for t in trades:
        rk = risk_map.get((t.symbol, t.entry_time))
        if rk is not None:
            t.risk_executed = True
            t.risk_acct_ret_pct = rk.acct_ret_pct
            t.risk_after_usd = rk.equity_after_usd
            last_risk_eq = rk.equity_after_usd
        else:
            t.risk_executed = False
            t.risk_acct_ret_pct = 0.0
            t.risk_after_usd = last_risk_eq

        rl = roll_map.get((t.symbol, t.entry_time))
        if rl is not None:
            t.roll_executed = True
            t.roll_ret_pct = rl.acct_ret_pct
            t.roll_after_usd = rl.equity_after_usd
            last_roll_eq = rl.equity_after_usd
        else:
            t.roll_executed = False
            t.roll_ret_pct = 0.0
            t.roll_after_usd = last_roll_eq

    return {
        "init_cap": init_cap,
        "leverage": leverage,
        "risk_pct_per_trade": risk_pct_per_trade,
        "max_positions": max_positions,
        "final_risk_usd": hr["final_eq"],
        "risk_ret_pct": hr["ret_pct"],
        "risk_mdd_pct": hr["mdd_pct"],
        "risk_executed": hr["executed_count"],
        "risk_skipped": hr["skipped_count"],
        "risk_curve": hr["curve_values"],
        "risk_times": hr["curve_times"],
        "final_spot_1x_usd": sp["final_eq"],
        "spot_1x_ret_pct": sp["ret_pct"],
        "spot_1x_mdd_pct": sp["mdd_pct"],
        "spot_1x_curve": sp["curve_values"],
        "spot_1x_times": sp["curve_times"],
        "final_roll_usd": c3["final_eq"],
        "roll_ret_pct": c3["ret_pct"],
        "roll_mdd_pct": c3["mdd_pct"],
        "executed": c3["executed_count"],
        "skipped": c3["skipped_count"],
        "liq_capped_3x": c3["liquidity_capped_count"],
        "liquidated_3x": c3["liquidated_count"],
        "roll_curve": c3["curve_values"],
        "roll_times": c3["curve_times"],
        "final_fixed_usd": f3["final_eq"],
        "fixed_ret_pct": f3["ret_pct"],
        "fixed_mdd_pct": f3["mdd_pct"],
        "fixed_curve": f3["curve_values"],
        "arith_sum_1x_pct": res["arith_sum_1x_pct"],
        "total_funding_pct": res["total_funding_pct"],
    }


def plot_equity_curve(trades: Sequence[TradeResult], out_path: Path, roll_stats: Dict[str, Any]) -> None:
    """繪製誠實資金淨值曲線（主圖：2% 以損定倉複利 + 1x 真實複利 + 3x 含強平/流動性上限對照）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if not trades:
        return

    init_cap = roll_stats["init_cap"]
    lev = roll_stats["leverage"]
    rpct = roll_stats["risk_pct_per_trade"]

    fig, (ax_top, ax_bot) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.2),
        facecolor="#ffffff",
        gridspec_kw={"wspace": 0.18},
    )

    if roll_stats.get("risk_times"):
        ax_top.step(
            roll_stats["risk_times"],
            roll_stats["risk_curve"],
            where="post",
            color="#00897b",
            linewidth=2.5,
            label=f"★ 主指標 {rpct:g}% 以損定倉複利 (${roll_stats['final_risk_usd']:,.0f} / {roll_stats['risk_ret_pct']:+.1f}%)",
        )
    if roll_stats.get("spot_1x_times"):
        ax_top.step(
            roll_stats["spot_1x_times"],
            roll_stats["spot_1x_curve"],
            where="post",
            color="#1e88e5",
            linewidth=1.9,
            linestyle="--",
            label=f"1x 真實無槓桿複利 (${roll_stats['final_spot_1x_usd']:,.0f} / {roll_stats['spot_1x_ret_pct']:+.1f}%)",
        )
    if roll_stats.get("roll_times"):
        ax_top.step(
            roll_stats["roll_times"],
            roll_stats["roll_curve"],
            where="post",
            color="#d50000",
            linewidth=1.6,
            linestyle=":",
            label=f"{lev:g}x 滾倉 [含強平/流動性上限·理論] (${roll_stats['final_roll_usd']:,.0f} / 強平{roll_stats['liquidated_3x']}次)",
        )
    ax_top.set_yscale("log")
    ax_top.axhline(init_cap, color="#263238", linewidth=0.9, linestyle=":")
    ax_top.set_title(f"${init_cap:,.0f} 起誠實帳戶淨值曲線 (對數刻度, 已扣費滑+資金費率)", fontsize=11.5, fontweight="bold")
    ax_top.set_ylabel("帳戶餘額 (USD)", fontsize=10)
    ax_top.set_xlabel("平倉時間推進", fontsize=10)
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=8.6, framealpha=0.9)

    # 右圖：1R = 2% 單利標準曲線 (R倍數)
    eq_r = np.r_[0.0, np.cumsum([t.r_ma25 for t in trades])]
    ax_bot.plot(
        range(len(eq_r)),
        eq_r,
        color="#1e88e5",
        linewidth=2.2,
        label=f"策略純盈虧比累積 (總獲利: {eq_r[-1]:+.1f}R)",
    )
    ax_bot.fill_between(range(len(eq_r)), 0, eq_r, color="#e3f2fd", alpha=0.55)
    ax_bot.axhline(0, color="#263238", linewidth=0.9, linestyle=":")
    ax_bot.set_title("無槓桿純策略盈虧比曲線 (R 倍數)", fontsize=12, fontweight="bold")
    ax_bot.set_ylabel("累積盈虧比 (R)", fontsize=10)
    ax_bot.set_xlabel("交易筆數推進", fontsize=10)
    ax_bot.grid(True, linestyle="--", alpha=0.3)
    ax_bot.legend(loc="upper left", fontsize=9.2, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _engine_sub_stats(sub_trades: Sequence[TradeResult]) -> Dict[str, float]:
    if not sub_trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "total_r": 0.0, "total_pct": 0.0, "pf": 0.0}
    rs = np.array([t.r_ma25 for t in sub_trades], dtype=float)
    pcts = np.array([t.pnl_pct_ma25 for t in sub_trades], dtype=float)
    wins = int(np.sum(rs > 0))
    gw = float(np.sum(rs[rs > 0]))
    gl = abs(float(np.sum(rs[rs < 0])))
    pf = gw / gl if gl > 0 else 999.0
    return {
        "n": len(sub_trades),
        "wins": wins,
        "wr": float(wins / len(sub_trades) * 100.0),
        "total_r": float(np.sum(rs)),
        "total_pct": float(np.sum(pcts)),
        "pf": pf,
    }


def generate_html_report(
    trades: Sequence[TradeResult],
    featured_trades: Sequence[TradeResult],
    out_html: Path,
    interval: str,
    days: int,
    n_symbols: int,
    roll_stats: Dict[str, Any],
    oos_report: Optional[Dict[str, Any]] = None,
) -> None:
    """產出完整的 HTML 視覺化誠實戰績看板（支援 v8 雙引擎合體版 + IS/OOS 檢驗）。"""
    rs = np.array([t.r_ma25 for t in trades], dtype=float)
    n = len(trades)
    win_rate = float(np.mean(rs > 0) * 100.0) if n else 0.0
    gross_win = float(np.sum(rs[rs > 0])) if n else 0.0
    gross_loss = abs(float(np.sum(rs[rs < 0]))) if n else 1.0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0

    eng_a_trades = [t for t in trades if t.signal_type != "COILED_MA200_STAGE2"]
    eng_b_trades = [t for t in trades if t.signal_type == "COILED_MA200_STAGE2"]
    st_a = _engine_sub_stats(eng_a_trades)
    st_b = _engine_sub_stats(eng_b_trades)
    st_all = _engine_sub_stats(trades)
    rpct = roll_stats["risk_pct_per_trade"]
    oos_section_html = bt_utils.render_oos_walkforward_html(oos_report, risk_pct=rpct) if oos_report else ""

    win_cards_html = []
    loss_cards_html = []
    for t in featured_trades:
        if not t.chart_file:
            continue
        is_win = t.r_ma25 > 0
        badge_bg = "#00c853" if is_win else "#ff5252"
        oos_badge = " [OOS]" if t.is_oos else " [IS]"
        if t.signal_type == "BOTTOM_BREAKOUT":
            stype = f"★ 引擎A·底部爆量起漲{oos_badge}"
            tag_style = "background:#b71c1c;color:#fff;"
        elif t.signal_type == "COILED_MA200_STAGE2":
            stype = f"🐂 引擎B·牛來黏帶穿MA200二拍{oos_badge}"
            tag_style = "background:#6a1b9a;color:#fff;"
        else:
            stype = f"▲ 引擎A·回踩MA25二次起漲{oos_badge}"
            tag_style = "background:#006064;color:#fff;"
        risk_tag = (
            f"{rpct:g}%定倉帳戶: {t.risk_acct_ret_pct:+.2f}% → 淨值 ${t.risk_after_usd:,.0f}"
            if t.risk_executed
            else "滿倉重疊(略過)"
        )
        card_block = (
            f"""
            <div class="card">
              <div class="card-header">
                <div>
                  <span class="sym">{escape(t.symbol)}</span>
                  <span class="tag" style="{tag_style}">{stype}</span>
                  <span class="time">{t.entry_time.strftime('%Y-%m-%d %H:%M')} TPE</span>
                </div>
                <div class="pnl-badge" style="background:{badge_bg}">
                  1x淨利: {t.pnl_pct_ma25:+.1f}% ({t.r_ma25:+.2f}R)
                </div>
              </div>
              <div class="card-meta">
                進場: <b>{t.entry_price:.6g}</b> ｜ 初始止損: <b>{t.stop_price:.6g}</b> (風險 {t.risk_pct:.1f}%) ｜
                出場: <b>{t.exit_price:.6g}</b> ({escape(t.exit_reason_ma25)}) ｜ 費率: <b>{t.funding_cost_pct:+.2f}%</b> ｜ <b>{risk_tag}</b>
              </div>
              <div class="card-note">{escape(t.note)}</div>
              <a href="cards/{escape(t.chart_file)}" target="_blank">
                <img src="cards/{escape(t.chart_file)}" loading="lazy" alt="{escape(t.symbol)}"/>
              </a>
            </div>
            """
        )
        if is_win:
            win_cards_html.append(card_block)
        else:
            loss_cards_html.append(card_block)

    rows_html = []
    for idx, t in enumerate(reversed(trades), 1):
        color_cls = "pos" if t.r_ma25 > 0 else "neg"
        split_lbl = "OOS" if t.is_oos else "IS"
        if t.signal_type == "BOTTOM_BREAKOUT":
            stype = "★ 引擎A·底部起漲"
        elif t.signal_type == "COILED_MA200_STAGE2":
            stype = "🐂 引擎B·牛來二拍"
        else:
            stype = "▲ 引擎A·回踩MA25"
        acct_cell = f"{t.risk_acct_ret_pct:+.2f}%" if t.risk_executed else "滿倉略過"
        bal_cell = f"${t.risk_after_usd:,.2f}" if t.risk_executed else f"(${t.risk_after_usd:,.2f})"
        rows_html.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
              <td><b>{escape(t.symbol)}</b></td>
              <td>{split_lbl}</td>
              <td>{stype}</td>
              <td>{t.entry_price:.6g}</td>
              <td>{t.stop_price:.6g} ({t.risk_pct:.1f}%)</td>
              <td>{t.exit_price:.6g}</td>
              <td>{escape(t.exit_reason_ma25)}</td>
              <td>{t.funding_cost_pct:+.2f}%</td>
              <td class="{color_cls}"><b>{t.pnl_pct_ma25:+.1f}%</b></td>
              <td class="{color_cls}"><b>{acct_cell}</b></td>
              <td><b>{bal_cell}</b></td>
              <td class="{color_cls}"><b>{t.r_ma25:+.2f}R</b></td>
            </tr>
            """
        )

    top_a = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_pct_ma25:+.0f}%)" for t in sorted(eng_a_trades, key=lambda x: x.pnl_pct_ma25, reverse=True)[:5]) or "-"
    top_b = "、".join(f"{t.symbol.replace('USDT','')} ({t.pnl_pct_ma25:+.0f}%)" for t in sorted(eng_b_trades, key=lambda x: x.pnl_pct_ma25, reverse=True)[:5]) or "-"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安永續「v8 雙引擎合體版」{days}天誠實回測報告 ({interval})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Microsoft JhengHei", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1420px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom: 6px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 14px; margin-bottom: 22px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }}
  .kpi .label {{ font-size: 13px; color: #94a3b8; }}
  .kpi .val {{ font-size: 24px; font-weight: 800; margin-top: 6px; color: #38bdf8; }}
  .kpi .val.green {{ color: #4ade80; }}
  .kpi .val.amber {{ color: #fbbf24; }}
  .eq-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 28px; text-align: center; }}
  .eq-box img {{ max-width: 100%; border-radius: 8px; }}
  h2 {{ font-size: 20px; margin: 28px 0 14px; color: #f1f5f9; border-left: 4px solid #38bdf8; padding-left: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(580px, 1fr)); gap: 18px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .sym {{ font-size: 18px; font-weight: 800; color: #f8fafc; margin-right: 8px; }}
  .tag {{ background: #334155; color: #e2e8f0; padding: 3px 8px; border-radius: 6px; font-size: 12px; margin-right: 8px; font-weight: 600; }}
  .time {{ color: #94a3b8; font-size: 13px; }}
  .pnl-badge {{ color: #fff; font-weight: 800; padding: 4px 10px; border-radius: 8px; font-size: 14px; }}
  .card-meta {{ font-size: 13px; color: #cbd5e1; margin-bottom: 4px; }}
  .card-note {{ font-size: 12.5px; color: #94a3b8; margin-bottom: 10px; }}
  .card img {{ width: 100%; border-radius: 8px; border: 1px solid #334155; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; font-size: 13.5px; margin-bottom: 24px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
  .pos {{ color: #4ade80; }}
  .neg {{ color: #f87171; }}
</style>
</head>
<body>
<div class="container">
  <h1>🚀 幣安永續合約「v8 雙引擎合體版」過去 {days} 天誠實回測戰績看板</h1>
  <div class="sub">
    K線週期：<b>{escape(interval)}</b> ｜ 時點滾動成交額標的池：<b>{n_symbols} 檔純加密貨幣 USDT 永續合約（無強制塞幣、無未來成交額前視）</b> ｜
    已扣雙邊費滑 <b>{FEE_ROUNDTRIP_PCT}%</b> 與歷史/保守資金費率
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">★ 主指標：{rpct:g}% 以損定倉複利</div><div class="val green">${roll_stats['final_risk_usd']:,.2f} ({roll_stats['risk_ret_pct']:+,.1f}%)</div></div>
    <div class="kpi"><div class="label">{rpct:g}% 定倉最大回撤 (最多{roll_stats['max_positions']}倉)</div><div class="val" style="color:#f87171">-{roll_stats['risk_mdd_pct']:.1f}%</div></div>
    <div class="kpi"><div class="label">1x 真實無槓桿複利</div><div class="val green">${roll_stats['final_spot_1x_usd']:,.2f} ({roll_stats['spot_1x_ret_pct']:+,.1f}%)</div></div>
    <div class="kpi"><div class="label">雙引擎總勝率 ({n} 筆)</div><div class="val green">{win_rate:.1f}%</div></div>
    <div class="kpi"><div class="label">雙引擎獲利因子 (PF)</div><div class="val amber">{pf:.2f}</div></div>
    <div class="kpi"><div class="label">單筆報酬算術總和 (非複利)</div><div class="val">{st_all['total_pct']:+.1f}%</div></div>
    <div class="kpi"><div class="label">{roll_stats['leverage']:g}x 滾倉 [含強平/流動性上限·理論]</div><div class="val amber">${roll_stats['final_roll_usd']:,.0f} (-{roll_stats['roll_mdd_pct']:.1f}%)</div></div>
  </div>

  {oos_section_html}

  <h2>⚡ 雙引擎分項表現對照（0% 訊號重疊，互補增效）</h2>
  <table>
    <thead>
      <tr>
        <th>子引擎名稱</th><th>核心形態與出場邏輯</th><th>交易筆數</th><th>實戰勝率</th><th>單筆算術總和(非複利)</th><th>累積盈虧比 (R)</th><th>獲利因子 (PF)</th><th>代表標的</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>★ 引擎 A（平地爆量突破）</b></td>
        <td>8均線全收斂(&le;3.4%) + 底部黃金收斂/極窄箱體 + 10~34x 巨量突破 → 沿 15m MA25 移動停利</td>
        <td><b>{st_a['n']} 筆</b> ({st_a['wins']}勝)</td>
        <td class="pos"><b>{st_a['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_a['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_a['total_r']:+.1f}R</b></td>
        <td><b>{st_a['pf']:.2f}</b></td>
        <td>{escape(top_a)}</td>
      </tr>
      <tr>
        <td><b>🐂 引擎 B（牛來 · 水下穿MA200二拍）</b></td>
        <td>MA200水下築底&ge;80根 + 7線黏帶&le;1.8% + 第一拍2~8x穿MA200 + 第二拍紅K&ge;1%強確認 → 沿 MA99 (1H MA25) 移動停利</td>
        <td><b>{st_b['n']} 筆</b> ({st_b['wins']}勝)</td>
        <td class="pos"><b>{st_b['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_b['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_b['total_r']:+.1f}R</b></td>
        <td><b>{st_b['pf']:.2f}</b></td>
        <td>{escape(top_b)}</td>
      </tr>
      <tr style="background:#0f172a;">
        <td><b>🚀 雙引擎合體總計</b></td>
        <td>{rpct:g}% 以損定倉併發執行 {roll_stats['risk_executed']} 筆（滿倉略過 {roll_stats['risk_skipped']} 筆）｜3x 理論對照流動性受限 {roll_stats['liq_capped_3x']} 次、強平 {roll_stats['liquidated_3x']} 次</td>
        <td><b>{st_all['n']} 筆</b> ({st_all['wins']}勝)</td>
        <td class="pos"><b>{st_all['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_all['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_all['total_r']:+.1f}R</b></td>
        <td class="pos"><b>{st_all['pf']:.2f}</b></td>
        <td>主指標 {rpct:g}%定倉 <b>${roll_stats['final_risk_usd']:,.0f}</b>（1x複利 <b>${roll_stats['final_spot_1x_usd']:,.0f}</b>）</td>
      </tr>
    </tbody>
  </table>

  <div class="eq-box">
    <img src="equity_curve.png" alt="Equity Curve"/>
  </div>

  <h2>📈 獲利交易復盤圖卡（共 {len(win_cards_html)} 筆）</h2>
  <div class="grid">
    {''.join(win_cards_html)}
  </div>

  <h2>📉 虧損交易復盤圖卡（共 {len(loss_cards_html)} 筆）</h2>
  <div class="grid">
    {''.join(loss_cards_html)}
  </div>

  <h2>📋 過去 {days} 天完整交易明細表（共 {n} 筆，依時間由新到舊）</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>進場時間 (TPE)</th><th>幣種</th><th>區間</th><th>訊號類型</th>
        <th>進場價</th><th>止損價 (風險)</th><th>出場價</th><th>出場原因</th>
        <th>資金費率成本</th><th>1x淨利</th><th>{rpct:g}%定倉帳戶報酬</th><th>{rpct:g}%定倉後淨值</th><th>盈虧比 (R)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def summarize_trades(trades: Sequence[TradeResult], label: str) -> None:
    if not trades:
        print(f"[{label}] 無交易紀錄")
        return

    n = len(trades)
    for mode_name, r_attr in [
        ("A. 固定 3R 止盈出場", "r_fixed3r"),
        ("B. 雙引擎專屬移動停利 (A:MA25 / B:MA99)", "r_ma25"),
        ("C. 2R平半倉 + 移動停利", "r_hybrid"),
    ]:
        rs = np.array([getattr(t, r_attr) for t in trades], dtype=float)
        win_rate = float(np.mean(rs > 0) * 100.0)
        total_r = float(np.sum(rs))
        avg_r = float(np.mean(rs))
        gross_win = float(np.sum(rs[rs > 0]))
        gross_loss = abs(float(np.sum(rs[rs < 0])))
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

        eq_curve = np.cumsum(rs * 2.0)
        peaks = np.maximum.accumulate(np.r_[0.0, eq_curve])[:-1]
        max_dd = float(np.max(peaks - eq_curve)) if len(eq_curve) else 0.0

        print(
            f"  {mode_name:<36} | 交易數: {n:>4} | 勝率: {win_rate:>5.1f}% | "
            f"總獲利: {total_r:>+7.1f}R ({total_r*2.0:>+6.1f}%) | "
            f"期望值: {avg_r:>+5.2f}R/筆 | 獲利因子(PF): {pf:>5.2f} | 最大回撤: -{max_dd:.1f}%"
        )


def _ensure_ma_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df[~df.index.duplicated(keep="last")].sort_index().copy()
    for period, _, _, _ in scanner.MA_SPECS:
        if f"MA{period}" not in out.columns:
            out[f"MA{period}"] = out["Close"].rolling(period, min_periods=period).mean()
    if "MA240" not in out.columns:
        out["MA240"] = out["Close"].rolling(240, min_periods=240).mean()
    if "VOL_MA5" not in out.columns:
        out["VOL_MA5"] = out["Volume"].rolling(5, min_periods=3).mean()
    if "VOL_MA10" not in out.columns:
        out["VOL_MA10"] = out["Volume"].rolling(10, min_periods=5).mean()
    if "VOL_MA20" not in out.columns:
        out["VOL_MA20"] = out["Volume"].rolling(20, min_periods=20).mean()
    if "VOL_MA20_PREV" not in out.columns or "VOL_RATIO" not in out.columns:
        out["VOL_MA20_PREV"] = out["Volume"].shift(1).rolling(20, min_periods=10).mean()
        out["VOL_RATIO"] = out["Volume"] / out["VOL_MA20_PREV"].replace(0, np.nan)
    return out


def _npz_to_df(npz_path: Path) -> pd.DataFrame:
    z = np.load(npz_path)
    ts = pd.to_datetime(z["t"].astype(np.int64), unit="ms", utc=True).tz_convert(TPE)
    c_arr = z["c"].astype(float)
    v_arr = z["v"].astype(float)
    qv_arr = z["qv"].astype(float) if "qv" in z.files else (c_arr * v_arr)
    df = pd.DataFrame(
        {
            "Open": z["o"].astype(float),
            "High": z["h"].astype(float),
            "Low": z["l"].astype(float),
            "Close": c_arr,
            "Volume": v_arr,
            "QuoteVolume": qv_arr,
        },
        index=ts,
    )
    return _ensure_ma_columns(df)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="幣安永續 v8 雙引擎無偏誤誠實回測與 HTML 看板生成器")
    parser.add_argument("--interval", default="15m", help="15m or 1h")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top", type=int, default=0, help="時點滾動24h成交額前N名過濾 (0=全部符合min-vol之合約)")
    parser.add_argument("--min-vol", type=float, default=2_000_000, help="訊號當下前24h最低滾動成交額 (USDT)")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--risk-pct", type=float, default=2.0, help="主指標單筆風險百分比 (預設 2.0%%)")
    parser.add_argument("--max-positions", type=int, default=3, help="最大同時持倉數 (預設 3)")
    parser.add_argument("--funding-rate-8h", type=float, default=0.03, help="無法抓取歷史費率時的保守估計資金費率 (%%/8h)")
    parser.add_argument("--no-fetch-funding", action="store_true", help="不連網抓取歷史資金費率，直接使用保守估計值")
    parser.add_argument("--max-bar-vol-pct", type=float, default=0.02, help="單筆倉位佔突破K線成交額上限比例 (預設 0.02)")
    parser.add_argument("--no-liquidity-cap", action="store_true", help="停用倉位流動性上限")
    parser.add_argument("--oos-date", type=str, default="", help="樣本外盲測切分日期 YYYY-MM-DD")
    parser.add_argument("--oos-ratio", type=float, default=0.30, help="樣本外盲測時間佔比 (預設後 30%% 為 OOS)")
    parser.add_argument("--wf-folds", type=int, default=4, help="Walk-Forward 時間切片期數 (預設 4)")
    parser.add_argument("--cache-dir", default=os.environ.get("BINANCE_CACHE_DIR", "breakout_output/cache_365d"), help="本地 K 線快取路徑 (.pkl/.npz)")
    parser.add_argument("--npz-dir", default=os.environ.get("BINANCE_NPZ_DIR", ""), help="額外 NPZ 快取資料夾路徑")
    parser.add_argument("--html", action="store_true", help="相容旗標（預設皆產出 HTML 看板）")
    parser.add_argument("--skip-cards", action="store_true", help="略過單筆圖卡繪製")
    parser.add_argument("--out-dir", default="breakout_output/backtest_90d")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    npz_dir = Path(args.npz_dir) if args.npz_dir else None
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cards:
        for old_png in cards_dir.glob("*.png"):
            old_png.unlink()

    cached_dfs = bt_utils.load_cached_frames(
        [cache_dir, npz_dir, Path("breakout_output/cache_180d")],
        interval=args.interval,
        excluded_symbols=scanner.EXCLUDED_SYMBOLS,
        npz_converter=_npz_to_df,
    )

    now_dt = datetime.now(TPE)
    start_dt = now_dt - timedelta(days=args.days)
    if cached_dfs:
        all_universe_syms = sorted(cached_dfs.keys())
    else:
        all_universe_syms = bt_utils.list_historical_usdt_perps(
            scanner.SESSION,
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(now_dt.timestamp() * 1000),
            excluded_symbols=scanner.EXCLUDED_SYMBOLS,
            base_url=scanner.BASE,
        )

    print(f"正在對 {len(all_universe_syms)} 檔純加密貨幣永續合約過去 {args.days} 天 ({args.interval}) 執行 v8 雙引擎無偏誤誠實回測...")

    from collections import Counter
    raw_loaded: Dict[str, pd.DataFrame] = {}
    for sym in all_universe_syms:
        if sym in cached_dfs:
            df = _ensure_ma_columns(cached_dfs[sym])
        else:
            df = fetch_klines_paginated(sym, interval=args.interval, days=args.days, cache_dir=cache_dir)
        if df is None or df.empty:
            continue
        raw_loaded[sym] = df

    if not raw_loaded:
        print("無有效歷史 K 線資料可供回測。")
        return 0

    max_ts = max(df.index[-1] for df in raw_loaded.values())
    cutoff = max_ts.to_pydatetime() - timedelta(days=args.days)
    crop_dt = cutoff - timedelta(days=10)

    all_dfs: Dict[str, pd.DataFrame] = {}
    spike_counter: Counter = Counter()
    for sym, full_df in raw_loaded.items():
        df = full_df[full_df.index >= crop_dt]
        if len(df) <= 60:
            continue
        all_dfs[sym] = df
        vr = df["VOL_RATIO"].to_numpy(float)
        o = df["Open"].to_numpy(float)
        c = df["Close"].to_numpy(float)
        gain = (c - o) / np.maximum(o, 1e-9) * 100.0
        mask = (vr >= 5.0) & (gain >= 3.0)
        for ts in df.index[mask]:
            spike_counter[ts] += 1

    pit_index = bt_utils.PointInTimeVolumeIndex(all_dfs)
    funding_store = bt_utils.FundingRateStore(
        cache_dir=cache_dir,
        fallback_rate_8h_pct=args.funding_rate_8h,
        fetch_online=not args.no_fetch_funding,
        session=scanner.SESSION,
        base_url=scanner.BASE,
    )

    trades_a: List[TradeResult] = []
    trades_b: List[TradeResult] = []
    dfs_by_sym: Dict[str, pd.DataFrame] = {}

    for sym, df in sorted(all_dfs.items()):
        sigs_a = scanner.detect_signals(df, sym, args.interval)
        out_a: List[TradeResult] = []
        for s in sigs_a:
            if s.timestamp < cutoff or spike_counter[s.timestamp] >= 3:
                continue
            ok_vol, qv24, _ = pit_index.check_eligibility(
                sym, s.timestamp, min_quote_vol=args.min_vol, top_n=args.top
            )
            if not ok_vol:
                continue
            tr_a = simulate_trade(df, s, funding_store=funding_store, trailing_24h_qv_usd=qv24)
            if tr_a is not None:
                out_a.append(tr_a)
        if out_a:
            dfs_by_sym[sym] = df
            trades_a.extend(out_a)

        if len(df) > 245:
            sigs_b = scanner.detect_coiled_ma200_signals(df, sym, args.interval)
            out_b: List[TradeResult] = []
            for s in sigs_b:
                if s.timestamp < cutoff or spike_counter[s.timestamp] >= 5:
                    continue
                ok_vol, qv24, _ = pit_index.check_eligibility(
                    sym, s.timestamp, min_quote_vol=args.min_vol, top_n=args.top
                )
                if not ok_vol:
                    continue
                tr_b = simulate_coiled_trade(df, s, funding_store=funding_store, trailing_24h_qv_usd=qv24)
                if tr_b is not None:
                    out_b.append(tr_b)
            if out_b:
                dfs_by_sym[sym] = df
                trades_b.extend(out_b)

    ts_sig_counts = Counter(t.entry_time for t in trades_a)
    filtered_a = [t for t in trades_a if ts_sig_counts[t.entry_time] < 2]
    all_trades = sorted(filtered_a + trades_b, key=lambda t: t.entry_time)

    min_eval_ts = max(min(df.index[0] for df in all_dfs.values()).to_pydatetime(), cutoff)
    oos_report = bt_utils.compute_oos_and_walkforward(
        _to_portfolio_inputs(all_trades),
        window_start=min_eval_ts,
        window_end=max_ts.to_pydatetime(),
        oos_split_date=args.oos_date,
        oos_ratio=args.oos_ratio,
        wf_folds=args.wf_folds,
        initial_cap=args.capital,
        risk_pct_per_trade=args.risk_pct,
        max_positions=args.max_positions,
    )
    split_ts = oos_report["split_ts"]
    for t in all_trades:
        t.is_oos = pd.Timestamp(t.entry_time) >= split_ts

    roll_stats = compute_3x_rolling(
        all_trades,
        init_cap=args.capital,
        leverage=args.leverage,
        risk_pct_per_trade=args.risk_pct,
        max_positions=args.max_positions,
        apply_liquidity_cap=not args.no_liquidity_cap,
        max_bar_vol_share=args.max_bar_vol_pct,
    )

    ins_st = oos_report["in_sample"]
    oos_st = oos_report["out_of_sample"]
    print("\n" + "=" * 120)
    print(
        f"【v8 雙引擎合體版 — {args.days} 天誠實回測總結】"
        f"週期: {args.interval} | 標的池: {len(all_dfs)} 檔 | "
        f"引擎A: {len(filtered_a)} 筆 + 引擎B(牛來): {len(trades_b)} 筆 = 共 {len(all_trades)} 筆"
    )
    print("=" * 120)
    summarize_trades(all_trades, "雙引擎全部訊號合計")
    print("-" * 120)
    print(
        f"  ★ 主指標 ({args.risk_pct:g}% 以損定倉複利, 最多{args.max_positions}倉): "
        f"${roll_stats['final_risk_usd']:,.2f} ({roll_stats['risk_ret_pct']:+,.1f}%) | MDD: -{roll_stats['risk_mdd_pct']:.1f}%\n"
        f"  1x 真實無槓桿複利                            : ${roll_stats['final_spot_1x_usd']:,.2f} ({roll_stats['spot_1x_ret_pct']:+,.1f}%)\n"
        f"  {args.leverage:g}x 單倉滾倉 [含強平/流動性上限·理論對照]     : ${roll_stats['final_roll_usd']:,.2f} ({roll_stats['roll_ret_pct']:+,.1f}%) | "
        f"MDD: -{roll_stats['roll_mdd_pct']:.1f}% | 流動性受限 {roll_stats['liq_capped_3x']} 次 | 強平 {roll_stats['liquidated_3x']} 次\n"
        f"  [IS 樣本內 < {oos_report['split_str'][:10]}] {ins_st['n']} 筆 | 勝率 {ins_st['win_rate']:.1f}% | 總R {ins_st['sum_r']:+.1f}R | {args.risk_pct:g}%定倉 ${ins_st['risk_eq']:,.0f}\n"
        f"  [OOS 樣本外 >= {oos_report['split_str'][:10]}] {oos_st['n']} 筆 | 勝率 {oos_st['win_rate']:.1f}% | 總R {oos_st['sum_r']:+.1f}R | {args.risk_pct:g}%定倉 ${oos_st['risk_eq']:,.0f}"
    )
    print("=" * 120)

    plot_equity_curve(all_trades, out_dir / "equity_curve.png", roll_stats)

    winners = sorted([t for t in all_trades if t.r_ma25 > 0], key=lambda t: t.r_ma25, reverse=True)
    losers = [t for t in all_trades if t.r_ma25 <= 0]
    featured_map: Dict[str, TradeResult] = {}
    if not args.skip_cards:
        for t in winners + losers:
            key = f"{t.symbol}_{t.entry_time.strftime('%Y%m%d_%H%M')}"
            if key not in featured_map:
                t.chart_file = f"{key}.png"
                featured_map[key] = t
                if t.symbol in dfs_by_sym:
                    plot_single_trade_card(dfs_by_sym[t.symbol], t, cards_dir / t.chart_file)

    html_path = out_dir / "index.html"
    generate_html_report(
        all_trades,
        list(featured_map.values()),
        html_path,
        interval=args.interval,
        days=args.days,
        n_symbols=len(all_dfs),
        roll_stats=roll_stats,
        oos_report=oos_report,
    )
    print(f"\n[HTML 視覺化戰績看板已產生] {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
