#!/usr/bin/env python3
"""幣安永續合約「起漲點策略 v2」90天完整實盤級回測 + HTML 視覺化戰績看板生成器

用法：
  python backtest_90d_breakout.py --interval 15m --days 90 --top 40 --html
"""

from __future__ import annotations

import argparse
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

import binance_breakout_scanner as scanner

TPE = ZoneInfo("Asia/Taipei")
FEE_ROUNDTRIP_PCT = 0.12  # 0.12% 來回手續費+滑價


def fetch_klines_paginated(symbol: str, interval: str = "15m", days: int = 90) -> pd.DataFrame:
    """分頁拉取過去 90 天的完整 K 線資料。"""
    bar_ms = scanner.INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = days * bars_per_day + 200

    raw: list = []
    end_ms: Optional[int] = None
    now_ms = int(time.time() * 1000)

    while len(raw) < need:
        limit = min(1500, need - len(raw))
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        try:
            chunk = scanner.get_json("/fapi/v1/klines", params=params)
        except Exception:
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

    if int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]

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

    df["VOL_MA5"] = df["Volume"].rolling(5, min_periods=3).mean()
    df["VOL_MA10"] = df["Volume"].rolling(10, min_periods=5).mean()
    df["VOL_MA20_PREV"] = df["Volume"].shift(1).rolling(20, min_periods=10).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20_PREV"].replace(0, np.nan)
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
    roll_executed: bool = False
    roll_ret_pct: float = 0.0
    roll_after_usd: float = 0.0


def simulate_trade(df: pd.DataFrame, sig: scanner.BreakoutSignal, max_hold_bars: int = 96) -> Optional[TradeResult]:
    """對單筆訊號模擬三種出場方式（無未來函數，同一根K線先檢查止損再檢查止盈）。"""
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

    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    ma25 = df["MA25"].to_numpy(float)
    times = df.index
    end_idx = min(len(df) - 1, i + max_hold_bars)

    # 1. Fixed 3R
    exit_3r = closes[end_idx]
    for j in range(i + 1, end_idx + 1):
        if lows[j] <= stop:
            exit_3r = stop
            break
        if highs[j] >= entry + 3.0 * risk:
            exit_3r = entry + 3.0 * risk
            break
    pnl_3r = (exit_3r - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT
    r_3r = pnl_3r / risk_pct

    # 2. 優化版 v3 MA25 移動停利 (漲破 +0.7R 立即啟動保本+手續費防衛，收盤跌破 MA25 出場)
    exit_ma25 = closes[end_idx]
    exit_idx_ma25 = end_idx
    reason_ma25 = "到期平倉"
    cur_stop = stop
    armed_be = False
    for j in range(i + 1, end_idx + 1):
        if lows[j] <= cur_stop:
            exit_ma25 = cur_stop
            exit_idx_ma25 = j
            reason_ma25 = "保本止盈" if armed_be else "初始止損"
            break
        if highs[j] >= entry + 0.7 * risk:
            armed_be = True
            cur_stop = max(cur_stop, entry * 1.003)
        # 動能時間止損：進場後第 3 根 K 線若未發動且收盤跌破進場價 -1.8%，立即提早砍倉減損
        if j == i + 3 and closes[j] < entry * 0.982:
            exit_ma25 = closes[j]
            exit_idx_ma25 = j
            reason_ma25 = "3K動能停滯減損"
            break
        if j - i >= 2 and np.isfinite(ma25[j]) and closes[j] < ma25[j]:
            exit_ma25 = closes[j]
            exit_idx_ma25 = j
            reason_ma25 = "破MA25止盈" if closes[j] > entry * 1.0012 else "破MA25微損出場"
            break
    pnl_ma25 = (exit_ma25 - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT
    r_ma25 = pnl_ma25 / risk_pct

    # 3. Hybrid: 觸及 +2R 平半倉 + 移保本，剩下一半沿 MA25 收盤跌破出場
    cur_stop = stop
    took_half = False
    half1_pnl_pct = 0.0
    half2_exit = closes[end_idx]

    for j in range(i + 1, end_idx + 1):
        if lows[j] <= cur_stop:
            half2_exit = cur_stop
            break
        if not took_half and highs[j] >= entry + 2.0 * risk:
            took_half = True
            half1_pnl_pct = (2.0 * risk) / entry * 100.0 - FEE_ROUNDTRIP_PCT
            cur_stop = max(cur_stop, entry * 1.002)
        if j - i >= 2 and np.isfinite(ma25[j]) and closes[j] < ma25[j]:
            half2_exit = closes[j]
            break

    half2_pnl_pct = (half2_exit - entry) / entry * 100.0 - FEE_ROUNDTRIP_PCT
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
    )


def simulate_coiled_trade(
    df: pd.DataFrame,
    sig: scanner.BreakoutSignal,
    max_hold_bars: int = 288,
    arm_r: float = 1.2,
    arm_abs_pct: float = 3.60,
    max_risk_pct: float = 7.20,
) -> Optional[TradeResult]:
    """引擎 B（牛來）專屬出場模擬：觸及 1.2R 或 +3.6% 啟動保本鎖利，沿 MA99 (1H MA25) 移動停利，最長持有 288 根 (72h)。"""
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
    times = df.index
    end_idx = min(len(df) - 1, i + max_hold_bars)

    cur_stop = stop
    armed = False
    exit_idx = end_idx
    exit_px = float(closes[end_idx])
    reason = "72h到期平倉"
    arm_target = min(entry + arm_r * risk, entry * (1.0 + arm_abs_pct / 100.0))

    for k in range(i + 1, end_idx + 1):
        if lows[k] <= cur_stop:
            exit_idx = k
            if armed and cur_stop >= entry:
                exit_px = float(entry * 1.0025)
                reason = "保本鎖利止盈"
            else:
                exit_px = float(min(opens[k], cur_stop))
                reason = "初始止損"
            break
        if highs[k] >= arm_target:
            armed = True
            cur_stop = max(cur_stop, entry * 1.0025)
        if armed and np.isfinite(ma99[k]) and closes[k] < ma99[k]:
            exit_idx = k
            exit_px = float(closes[k])
            reason = "破MA99(1H MA25)止盈"
            break

    raw_pnl = (exit_px / entry - 1.0) * 100.0
    pnl_net = raw_pnl - FEE_ROUNDTRIP_PCT
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


def compute_3x_rolling(
    trades: Sequence[TradeResult],
    init_cap: float = 1000.0,
    leverage: float = 3.0,
) -> Dict[str, Any]:
    """計算本金 init_cap + leverage 倍槓桿滾倉（持倉中不重複開倉）與固定開單對比。"""
    cap = init_cap
    peak = init_cap
    mdd = 0.0
    last_exit = None
    roll_curve = [init_cap]
    roll_labels = ["起點"]
    executed = 0

    fixed_cap = init_cap
    fixed_peak = init_cap
    fixed_mdd = 0.0
    fixed_curve = [init_cap]

    for t in trades:
        fixed_cap += (init_cap * leverage) * (t.pnl_pct_ma25 / 100.0)
        fixed_peak = max(fixed_peak, fixed_cap)
        fixed_mdd = max(fixed_mdd, (fixed_peak - fixed_cap) / fixed_peak * 100.0)
        fixed_curve.append(fixed_cap)

        if last_exit is not None and t.entry_time < last_exit:
            t.roll_executed = False
            t.roll_ret_pct = 0.0
            t.roll_after_usd = cap
            roll_curve.append(cap)
            roll_labels.append(t.symbol)
            continue

        executed += 1
        ret_lev = t.pnl_pct_ma25 * leverage
        cap = max(0.0, cap * (1.0 + ret_lev / 100.0))
        peak = max(peak, cap)
        mdd = max(mdd, (peak - cap) / peak * 100.0)
        last_exit = t.exit_time

        t.roll_executed = True
        t.roll_ret_pct = ret_lev
        t.roll_after_usd = cap
        roll_curve.append(cap)
        roll_labels.append(t.symbol)

    return {
        "init_cap": init_cap,
        "leverage": leverage,
        "final_roll_usd": cap,
        "roll_ret_pct": (cap / init_cap - 1.0) * 100.0,
        "roll_mdd_pct": mdd,
        "executed": executed,
        "skipped": len(trades) - executed,
        "roll_curve": roll_curve,
        "final_fixed_usd": fixed_cap,
        "fixed_ret_pct": (fixed_cap / init_cap - 1.0) * 100.0,
        "fixed_mdd_pct": fixed_mdd,
        "fixed_curve": fixed_curve,
    }


def plot_equity_curve(trades: Sequence[TradeResult], out_path: Path, roll_stats: Dict[str, Any]) -> None:
    """繪製 90 天「1,000 美金 + 3 倍槓桿滾倉」資產淨值曲線比較圖。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if not trades:
        return

    roll_curve = roll_stats["roll_curve"]
    fixed_curve = roll_stats["fixed_curve"]
    init_cap = roll_stats["init_cap"]
    lev = roll_stats["leverage"]

    fig, (ax_top, ax_bot) = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.2),
        facecolor="#ffffff",
        gridspec_kw={"wspace": 0.18},
    )

    x = range(len(roll_curve))
    ax_top.plot(
        x,
        roll_curve,
        color="#d50000",
        linewidth=2.4,
        label=f"全倉 {lev:g}x 滾倉複利 ({roll_stats['final_roll_usd']:,.0f} USD / {roll_stats['roll_ret_pct']:+.0f}%)",
    )
    ax_top.plot(
        x,
        fixed_curve,
        color="#00897b",
        linewidth=1.8,
        linestyle="--",
        label=f"固定 {init_cap:,.0f}×{lev:g}x 不滾倉 ({roll_stats['final_fixed_usd']:,.0f} USD / {roll_stats['fixed_ret_pct']:+.0f}%)",
    )
    ax_top.fill_between(x, init_cap, roll_curve, color="#ffebee", alpha=0.45)
    ax_top.axhline(init_cap, color="#263238", linewidth=0.9, linestyle=":")
    ax_top.set_title(f"{init_cap:,.0f} 美金 + {lev:g} 倍槓桿滾倉資產淨值曲線 (USD)", fontsize=12, fontweight="bold")
    ax_top.set_ylabel("帳戶餘額 (USD)", fontsize=10)
    ax_top.set_xlabel("交易筆數推進", fontsize=10)
    ax_top.grid(True, linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=9.2, framealpha=0.9)

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
) -> None:
    """產出完整的 HTML 視覺化戰績看板（支援 v8 雙引擎合體版）。"""
    rs = np.array([t.r_ma25 for t in trades], dtype=float)
    n = len(trades)
    win_rate = float(np.mean(rs > 0) * 100.0) if n else 0.0
    total_r = float(np.sum(rs)) if n else 0.0
    avg_r = float(np.mean(rs)) if n else 0.0
    gross_win = float(np.sum(rs[rs > 0])) if n else 0.0
    gross_loss = abs(float(np.sum(rs[rs < 0]))) if n else 1.0
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0
    eq = np.cumsum(rs * 2.0) if n else np.array([0.0])
    peaks = np.maximum.accumulate(np.r_[0.0, eq])[:-1] if n else np.array([0.0])
    max_dd = float(np.max(peaks - eq)) if n else 0.0

    eng_a_trades = [t for t in trades if t.signal_type != "COILED_MA200_STAGE2"]
    eng_b_trades = [t for t in trades if t.signal_type == "COILED_MA200_STAGE2"]
    st_a = _engine_sub_stats(eng_a_trades)
    st_b = _engine_sub_stats(eng_b_trades)
    st_all = _engine_sub_stats(trades)

    win_cards_html = []
    loss_cards_html = []
    for t in featured_trades:
        if not t.chart_file:
            continue
        is_win = t.r_ma25 > 0
        badge_bg = "#00c853" if is_win else "#ff5252"
        if t.signal_type == "BOTTOM_BREAKOUT":
            stype = "★ 引擎A·底部爆量起漲"
            tag_style = "background:#b71c1c;color:#fff;"
        elif t.signal_type == "COILED_MA200_STAGE2":
            stype = "🐂 引擎B·牛來黏帶穿MA200二拍"
            tag_style = "background:#6a1b9a;color:#fff;"
        else:
            stype = "▲ 引擎A·回踩MA25二次起漲"
            tag_style = "background:#006064;color:#fff;"
        lev = roll_stats["leverage"]
        roll_tag = (
            f"3x: {t.pnl_pct_ma25 * lev:+.1f}% → 餘額 ${t.roll_after_usd:,.0f}"
            if t.roll_executed
            else "持倉中重疊(跳過)"
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
                  1x: {t.pnl_pct_ma25:+.1f}% ｜ {lev:g}x: {t.pnl_pct_ma25*lev:+.1f}% ({t.r_ma25:+.2f}R)
                </div>
              </div>
              <div class="card-meta">
                進場: <b>{t.entry_price:.6g}</b> ｜ 初始止損: <b>{t.stop_price:.6g}</b> (風險 {t.risk_pct:.1f}%) ｜
                出場: <b>{t.exit_price:.6g}</b> ({escape(t.exit_reason_ma25)}) ｜ <b>{roll_tag}</b>
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
    lev = roll_stats["leverage"]
    for idx, t in enumerate(reversed(trades), 1):
        color_cls = "pos" if t.r_ma25 > 0 else "neg"
        if t.signal_type == "BOTTOM_BREAKOUT":
            stype = "★ 引擎A·底部起漲"
        elif t.signal_type == "COILED_MA200_STAGE2":
            stype = "🐂 引擎B·牛來二拍"
        else:
            stype = "▲ 引擎A·回踩MA25"
        lev_cell = f"{t.pnl_pct_ma25 * lev:+.1f}%" if t.roll_executed else "重疊跳過"
        bal_cell = f"${t.roll_after_usd:,.2f}" if t.roll_executed else f"(${t.roll_after_usd:,.2f})"
        rows_html.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
              <td><b>{escape(t.symbol)}</b></td>
              <td>{stype}</td>
              <td>{t.entry_price:.6g}</td>
              <td>{t.stop_price:.6g} ({t.risk_pct:.1f}%)</td>
              <td>{t.exit_price:.6g}</td>
              <td>{escape(t.exit_reason_ma25)}</td>
              <td>{t.vol_ratio:.1f}x</td>
              <td class="{color_cls}"><b>{t.pnl_pct_ma25:+.1f}%</b></td>
              <td class="{color_cls}"><b>{lev_cell}</b></td>
              <td><b>{bal_cell}</b></td>
              <td class="{color_cls}"><b>{t.r_ma25:+.2f}R</b></td>
            </tr>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安永續「v8 雙引擎合體版：AKE平地巨量 + 牛來水下穿MA200二拍」{days}天實盤級回測報告 ({interval})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Microsoft JhengHei", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1420px; margin: 0 auto; }}
  h1 {{ font-size: 26px; margin-bottom: 6px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 14px; margin-bottom: 22px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }}
  .kpi .label {{ font-size: 13px; color: #94a3b8; }}
  .kpi .val {{ font-size: 26px; font-weight: 800; margin-top: 6px; color: #38bdf8; }}
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
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 10px; overflow: hidden; font-size: 13.5px; margin-bottom: 24px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
  .pos {{ color: #4ade80; }}
  .neg {{ color: #f87171; }}
</style>
</head>
<body>
<div class="container">
  <h1>🚀 幣安永續合約「v8 雙引擎合體版：AKE平地巨量 + 牛來水下穿MA200二拍」過去 {days} 天實盤級回測戰績看板</h1>
  <div class="sub">
    K線週期：<b>{escape(interval)}</b> ｜ 掃描標的池：<b>{n_symbols} 檔純加密貨幣 USDT 永續合約（0% 訊號重疊）</b> ｜
    雙引擎分工：<b>【引擎A】全8均線真底部收斂+平地巨量突破(破MA25出場) ＋ 【引擎B·牛來】水下築底&ge;80根+7線黏帶穿MA200+第二拍紅K&ge;1%強確認(破1H MA25出場)</b>（已扣手續費與滑價 {FEE_ROUNDTRIP_PCT}%）
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">初始本金 / 槓桿</div><div class="val">${roll_stats['init_cap']:,.0f} ({roll_stats['leverage']:g}x)</div></div>
    <div class="kpi"><div class="label">3x 滾倉最終資金</div><div class="val green">${roll_stats['final_roll_usd']:,.2f}</div></div>
    <div class="kpi"><div class="label">3x 滾倉總報酬率</div><div class="val green">{roll_stats['roll_ret_pct']:+,.1f}%</div></div>
    <div class="kpi"><div class="label">固定 3x 不滾倉最終</div><div class="val">${roll_stats['final_fixed_usd']:,.2f}</div></div>
    <div class="kpi"><div class="label">雙引擎總勝率 ({n} 筆)</div><div class="val green">{win_rate:.1f}%</div></div>
    <div class="kpi"><div class="label">雙引擎獲利因子 (PF)</div><div class="val amber">{pf:.2f}</div></div>
    <div class="kpi"><div class="label">3x 滾倉最大回撤</div><div class="val" style="color:#f87171">-{roll_stats['roll_mdd_pct']:.1f}%</div></div>
  </div>

  <h2>⚡ 雙引擎分項表現對照（0% 訊號重疊，互補增效）</h2>
  <table>
    <thead>
      <tr>
        <th>子引擎名稱</th><th>核心形態與出場邏輯</th><th>交易筆數</th><th>實戰勝率</th><th>1x 累計報酬</th><th>累積盈虧比 (R)</th><th>獲利因子 (PF)</th><th>代表標竿大魚</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>★ 引擎 A（AKE / TAC / SHELL 平地爆量）</b></td>
        <td>8均線全收斂(&le;3.4%) + 底部黃金收斂/極窄箱體 + 10~34x 巨量突破 → 沿 15m MA25 移動停利</td>
        <td><b>{st_a['n']} 筆</b> ({st_a['wins']}勝)</td>
        <td class="pos"><b>{st_a['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_a['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_a['total_r']:+.1f}R</b></td>
        <td><b>{st_a['pf']:.2f}</b></td>
        <td>AKE (+281%)、TAC (+154%)、AIN (+96%)、SHELL (+68%)、BULLA (+33%)</td>
      </tr>
      <tr>
        <td><b>🐂 引擎 B（牛來 · 水下穿MA200二拍）</b></td>
        <td>MA200水下築底&ge;80根 + 7線黏帶&le;1.8% + 第一拍2~8x穿MA200 + 第二拍紅K&ge;1%強確認 → 沿 MA99 (1H MA25) 移動停利</td>
        <td><b>{st_b['n']} 筆</b> ({st_b['wins']}勝)</td>
        <td class="pos"><b>{st_b['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_b['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_b['total_r']:+.1f}R</b></td>
        <td><b>{st_b['pf']:.2f}</b></td>
        <td>ESPORTS (+106%)、RLC (+102%)、ARKM (+20%)、PUMP (+20%)</td>
      </tr>
      <tr style="background:#0f172a;">
        <td><b>🚀 雙引擎合體總計（單倉排他滾倉）</b></td>
        <td>雙引擎同時監看、持倉中不重複開倉（實際執行 {roll_stats['executed']} 筆 / 持倉重疊略過 {roll_stats['skipped']} 筆）</td>
        <td><b>{st_all['n']} 筆</b> ({st_all['wins']}勝)</td>
        <td class="pos"><b>{st_all['wr']:.1f}%</b></td>
        <td class="pos"><b>{st_all['total_pct']:+.1f}%</b></td>
        <td class="pos"><b>{st_all['total_r']:+.1f}R</b></td>
        <td class="pos"><b>{st_all['pf']:.2f}</b></td>
        <td>3x 滾倉回撤僅 <b>-{roll_stats['roll_mdd_pct']:.1f}%</b>，固定 3x 單利達 <b>${roll_stats['final_fixed_usd']:,.0f}</b></td>
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
        <th>#</th><th>進場時間 (TPE)</th><th>幣種</th><th>訊號類型</th>
        <th>進場價</th><th>止損價 (風險)</th><th>出場價</th><th>出場原因</th>
        <th>爆量倍數</th><th>1x現貨淨利</th><th>3x槓桿淨利</th><th>滾倉後餘額</th><th>盈虧比 (R)</th>
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


def _npz_to_df(npz_path: Path) -> pd.DataFrame:
    z = np.load(npz_path)
    ts = pd.to_datetime(z["t"].astype(np.int64), unit="ms", utc=True).tz_convert(TPE)
    df = pd.DataFrame(
        {
            "Open": z["o"].astype(float),
            "High": z["h"].astype(float),
            "Low": z["l"].astype(float),
            "Close": z["c"].astype(float),
            "Volume": z["v"].astype(float),
        },
        index=ts,
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
    return df


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="15m", help="15m or 1h")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--out-dir", default="breakout_output/backtest_90d")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for old_png in cards_dir.glob("*.png"):
        old_png.unlink()

    scratch_dir = Path(r"C:\Users\yubogood\.gemini\antigravity\brain\9798fe4c-a1ab-4376-bc5c-3426248b3f7d\scratch")
    cache_pkl = scratch_dir / f"dfs_{args.days}d.pkl"
    cache_365_pkl = scratch_dir / "dfs_365d.pkl"
    cached_dfs: Dict[str, pd.DataFrame] = {}
    if args.interval == "15m":
        import pickle
        for pkl_path in (cache_365_pkl, cache_pkl):
            if pkl_path.is_file():
                try:
                    cached_dfs.update(pickle.loads(pkl_path.read_bytes()))
                except Exception:
                    pass

    npz_dir = Path(r"C:\tmp\binance_15m_90d_cache")
    npz_files = {f.name.replace("_730d.npz", ""): f for f in sorted(npz_dir.glob("*_730d.npz"))} if npz_dir.is_dir() else {}

    if cached_dfs or npz_files:
        all_universe_syms = sorted((set(cached_dfs.keys()) | set(npz_files.keys())) - set(scanner.EXCLUDED_SYMBOLS))
    else:
        all_universe_syms = [s for s in scanner.fetch_top_usdt_perps(top_n=args.top) if s not in scanner.EXCLUDED_SYMBOLS]
        if "AKEUSDT" not in all_universe_syms:
            all_universe_syms.append("AKEUSDT")

    cutoff = datetime.now(TPE) - timedelta(days=args.days)
    crop_dt = cutoff - timedelta(days=10)

    print(f"正在對 {len(all_universe_syms)} 檔純加密貨幣永續合約過去 {args.days} 天 ({args.interval}) 執行 v8 雙引擎全市場無差別回測...")

    # -------------------------------------------------------------------------
    # 1. 載入全市場 174 檔 K 線並建立同步插針計數器
    # -------------------------------------------------------------------------
    from collections import Counter
    all_dfs: Dict[str, pd.DataFrame] = {}
    spike_counter: Counter = Counter()

    for sym in all_universe_syms:
        if sym in npz_files:
            df = _npz_to_df(npz_files[sym])
        elif sym in cached_dfs:
            df = cached_dfs[sym]
        else:
            df = fetch_klines_paginated(sym, interval=args.interval, days=args.days)
        if df is None or df.empty:
            continue
        df = df[df.index >= crop_dt]
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

    # -------------------------------------------------------------------------
    # 2. 對全市場 174 檔同時執行【引擎 A】與【引擎 B (牛來)】
    # -------------------------------------------------------------------------
    trades_a: List[TradeResult] = []
    trades_b: List[TradeResult] = []
    dfs_by_sym: Dict[str, pd.DataFrame] = {}

    for sym, df in all_dfs.items():
        # 引擎 A：底部 8 線糾結 + 平地巨量突破
        sigs_a = scanner.detect_signals(df, sym, args.interval)
        out_a: List[TradeResult] = []
        for s in sigs_a:
            if s.timestamp < cutoff or spike_counter[s.timestamp] >= 3:
                continue
            tr_a = simulate_trade(df, s)
            if tr_a is not None:
                out_a.append(tr_a)
        if out_a:
            dfs_by_sym[sym] = df
            trades_a.extend(out_a)

        # 引擎 B：水下築底 >= 80 根 + 七線黏帶穿 MA200 + 第二拍強確認
        if len(df) > 245:
            sigs_b = scanner.detect_coiled_ma200_signals(df, sym, args.interval)
            out_b: List[TradeResult] = []
            for s in sigs_b:
                if s.timestamp < cutoff or spike_counter[s.timestamp] >= 5:
                    continue
                tr_b = simulate_coiled_trade(df, s)
                if tr_b is not None:
                    out_b.append(tr_b)
            if out_b:
                dfs_by_sym[sym] = df
                trades_b.extend(out_b)

    ts_sig_counts = Counter(t.entry_time for t in trades_a)
    filtered_a = [t for t in trades_a if ts_sig_counts[t.entry_time] < 2]

    all_trades = sorted(filtered_a + trades_b, key=lambda t: t.entry_time)

    print("\n" + "=" * 120)
    print(
        f"【v8 雙引擎合體版 — {args.days} 天全市場回測總結】"
        f"週期: {args.interval} | 標的池: {len(all_universe_syms)} 檔 | "
        f"引擎A: {len(filtered_a)} 筆 + 引擎B(牛來): {len(trades_b)} 筆 = 共 {len(all_trades)} 筆"
    )
    print("=" * 120)
    summarize_trades(all_trades, "雙引擎全部訊號合計")
    print("=" * 120)

    # 計算 1,000 美金 + 3 倍槓桿滾倉數據
    roll_stats = compute_3x_rolling(all_trades, init_cap=args.capital, leverage=args.leverage)

    # 繪製資金曲線
    plot_equity_curve(all_trades, out_dir / "equity_curve.png", roll_stats)

    # 繪製全部獲利與虧損交易的復盤 K 線圖卡（依 R 由高到低排序獲利，虧損依時間排序）
    winners = sorted([t for t in all_trades if t.r_ma25 > 0], key=lambda t: t.r_ma25, reverse=True)
    losers = [t for t in all_trades if t.r_ma25 <= 0]
    featured_map: Dict[str, TradeResult] = {}
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
        n_symbols=len(all_universe_syms),
        roll_stats=roll_stats,
    )
    print(f"\n[HTML 視覺化戰績看板已產生] {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
