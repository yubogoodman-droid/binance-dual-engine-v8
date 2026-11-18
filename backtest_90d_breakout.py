#!/usr/bin/env python3
"""幣安永續合約「結構雙引擎二拍突破」無前視偏差實盤級回測 + HTML 視覺化戰績看板生成器

修正項目（針對原版 v8 審計問題全面重構）：
1. 移除私人硬碟黑箱路徑與寫死塞入的 AKEUSDT，支援本機標準快取 (--cache-dir) 或即時 API 分頁拉取。
2. 消除同根 K 線「幽靈保本成交」與止損無滑價 Bug：
   - 所有止損/保本觸發強制採 min(Open, StopPrice) 扣除跳空滑價。
   - 保本鎖利僅在「K 線收盤價」正式收過 +1.5R 後才於下一根生效，杜絕同根插針先衝高又跌破的假保本。
3. 真實交易摩擦：預設扣除 0.24% 來回手續費+滑價，並按實際持倉時間計入每 8 小時 0.01% 永續資金費率。
4. 訊號當下強制驗證「前 24h 真實滾動成交額 >= 500 萬 USDT」、BTC 急殺保護與同根集體插針過濾（與實盤掃描器 100% 同源）。
5. 廢除必爆倉的全倉 3x 滾倉神話，改採「單筆固定風險預算 (預設 1.5% 帳戶風險，A/B 雙引擎各最多 2 倉、單倉上限 2x 槓桿)」事件驅動資金管理，並同框對照前 275 天與近 90 天表現。

用法：
  python backtest_90d_breakout.py --interval 15m --days 90 --top 180 --html
"""

from __future__ import annotations

import argparse
import subprocess
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
FEE_ROUNDTRIP_PCT = 0.24     # 0.10% Taker 來回手續費 + 0.14% 雙邊實戰滑價
FUNDING_PER_8H_PCT = 0.01    # 永續合約每 8 小時基準資金費率拖累 (0.01%)


def fetch_klines_paginated(symbol: str, interval: str = "15m", days: int = 90) -> pd.DataFrame:
    """分頁拉取過去 N 天的完整已收盤 K 線資料。"""
    bar_ms = scanner.INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = (days + 12) * bars_per_day + 240

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
        time.sleep(0.03)

    if not raw:
        return pd.DataFrame()

    if int(raw[-1][0]) + bar_ms > now_ms:
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
    return scanner.enrich_indicators(df, interval=interval)


def load_cached_npz(npz_path: Path, interval: str = "15m") -> pd.DataFrame:
    """從標準壓縮檔載入 K 線 (支援 data=[t,o,h,l,c,v,qv] 或獨立欄位)。"""
    z = np.load(npz_path)
    if "data" in z:
        arr = z["data"]
        raw_t = arr[:, 0].astype(np.int64)
        ts_ms = np.where(raw_t > 10**14, raw_t // 1000, raw_t)
        df = pd.DataFrame(
            {
                "Open": arr[:, 1].astype(float),
                "High": arr[:, 2].astype(float),
                "Low": arr[:, 3].astype(float),
                "Close": arr[:, 4].astype(float),
                "Volume": arr[:, 5].astype(float),
                "QuoteVolume": arr[:, 6].astype(float),
            },
            index=pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(TPE),
        )
    else:
        raw_t = z["t"].astype(np.int64)
        ts_ms = np.where(raw_t > 10**14, raw_t // 1000, raw_t)
        c = z["c"].astype(float)
        v = z["v"].astype(float)
        qv = z["qv"].astype(float) if "qv" in z else c * v
        df = pd.DataFrame(
            {
                "Open": z["o"].astype(float),
                "High": z["h"].astype(float),
                "Low": z["l"].astype(float),
                "Close": c,
                "Volume": v,
                "QuoteVolume": qv,
            },
            index=pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(TPE),
        )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return scanner.enrich_indicators(df, interval=interval)


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
    prev_24h_quote_vol: float
    fee_pct: float
    funding_pct: float
    pnl_pct_net: float
    r_net: float
    pnl_pct_fixed3r: float
    r_fixed3r: float
    exit_reason: str
    hold_bars: int
    note: str
    chart_file: str = ""
    executed: bool = False
    pos_leverage: float = 0.0
    acct_ret_pct: float = 0.0
    after_equity_usd: float = 0.0


def simulate_structural_trade(
    df: pd.DataFrame,
    sig: scanner.BreakoutSignal,
    *,
    max_hold_bars: int = 240,
    arm_r: float = 1.5,
    min_risk_pct: float = 1.0,
    max_risk_pct: float = 7.0,
    fee_roundtrip_pct: float = FEE_ROUNDTRIP_PCT,
    funding_per_8h_pct: float = FUNDING_PER_8H_PCT,
) -> Optional[TradeResult]:
    """雙引擎統一實盤級出場模擬（無同根幽靈保本、止損含跳空滑價、計入手續費滑價與資金費率）。"""
    i = sig.bar_idx
    if i >= len(df) - 2:
        return None

    entry = float(sig.entry_price)
    stop = float(sig.stop_price)
    risk = entry - stop
    if entry <= 0 or risk <= 0:
        return None

    risk_pct = risk / entry * 100.0
    if risk_pct < min_risk_pct or risk_pct > max_risk_pct:
        return None

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    ma99 = df["MA99"].to_numpy(float)
    times = df.index
    end_idx = min(len(df) - 1, i + max_hold_bars)

    bar_ms = scanner.INTERVAL_MS.get(sig.interval, 900_000)
    bars_per_8h = max(1.0, (8 * 3_600_000) / bar_ms)

    # 對照組：固定 3R 止盈出場（先檢查止損並扣除開盤跳空）
    exit_3r = float(closes[end_idx])
    exit_3r_idx = end_idx
    for k in range(i + 1, end_idx + 1):
        if lows[k] <= stop:
            exit_3r = float(min(opens[k], stop))
            exit_3r_idx = k
            break
        if highs[k] >= entry + 3.0 * risk:
            exit_3r = float(entry + 3.0 * risk)
            exit_3r_idx = k
            break
    fund_3r = ((exit_3r_idx - i) / bars_per_8h) * funding_per_8h_pct
    pnl_3r = (exit_3r / entry - 1.0) * 100.0 - fee_roundtrip_pct - fund_3r
    r_3r = pnl_3r / risk_pct

    # 主策略：收盤站上 +1.5R 後於次根啟動保本防護，並沿 MA99 (1H MA25) 收盤移動停利
    cur_stop = stop
    armed = False
    exit_idx = end_idx
    exit_px = float(closes[end_idx])
    reason = "60h到期平倉"

    for k in range(i + 1, end_idx + 1):
        # 1. 當根先以「前一根已生效」的 cur_stop 檢查低點觸價，並考慮開盤跳空滑價
        if lows[k] <= cur_stop:
            exit_idx = k
            exit_px = float(min(opens[k], cur_stop))
            reason = "保本鎖利止盈" if armed else "初始止損"
            break

        # 2. 收盤確認站上 +1.5R 後，才將止損上移至保本價（於 k+1 根起生效，徹底杜絕同根幽靈保本）
        if closes[k] >= entry + arm_r * risk:
            armed = True
            cur_stop = max(cur_stop, entry * 1.003)

        # 3. 啟動趨勢追蹤後，收盤跌破 MA99 (1H MA25) 趨勢線出場
        if armed and np.isfinite(ma99[k]) and closes[k] < ma99[k]:
            exit_idx = k
            exit_px = float(closes[k])
            reason = "破MA99趨勢止盈" if exit_px >= entry else "破MA99減損出場"
            break

    hold_bars = exit_idx - i
    funding_pct = (hold_bars / bars_per_8h) * funding_per_8h_pct
    raw_pnl = (exit_px / entry - 1.0) * 100.0
    pnl_net = raw_pnl - fee_roundtrip_pct - funding_pct
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
        prev_24h_quote_vol=sig.prev_24h_quote_vol,
        fee_pct=fee_roundtrip_pct,
        funding_pct=funding_pct,
        pnl_pct_net=pnl_net,
        r_net=r_net,
        pnl_pct_fixed3r=pnl_3r,
        r_fixed3r=r_3r,
        exit_reason=reason,
        hold_bars=hold_bars,
        note=sig.note,
    )


def compute_portfolio_simulation(
    trades: Sequence[TradeResult],
    init_cap: float = 1000.0,
    risk_per_trade_pct: float = 1.5,
    max_pos_leverage: float = 2.0,
    slots_per_engine: int = 2,
    naive_full_leverage: float = 3.0,
) -> Dict[str, Any]:
    """事件驅動投資組合模擬：對比「1.5% 固定風險預算分倉」與「無腦 3x 全倉滾倉」的真實存活差異。"""
    ordered = sorted(
        trades,
        key=lambda t: (
            t.entry_time,
            0 if t.signal_type == "COILED_MA200_STAGE2" else 1,
            t.ma_spread_pct,
            -t.prev_24h_quote_vol,
        ),
    )

    equity = init_cap
    peak = init_cap
    mdd = 0.0
    eq_curve = [init_cap]
    open_positions: List[tuple[datetime, str, str, float, TradeResult]] = []
    executed_count = 0
    total_fee_drag_pct = 0.0
    total_funding_drag_pct = 0.0

    def _settle_until(cutoff_ts: Optional[datetime]) -> None:
        nonlocal equity, peak, mdd, open_positions
        still_open = []
        for ex_ts, eng_key, sym, pnl_usd, tr_ref in sorted(open_positions, key=lambda x: x[0]):
            if cutoff_ts is None or ex_ts <= cutoff_ts:
                equity = max(0.0, equity + pnl_usd)
                peak = max(peak, equity)
                if peak > 0:
                    mdd = max(mdd, (peak - equity) / peak * 100.0)
                tr_ref.after_equity_usd = equity
                eq_curve.append(equity)
            else:
                still_open.append((ex_ts, eng_key, sym, pnl_usd, tr_ref))
        open_positions = still_open

    for t in ordered:
        _settle_until(t.entry_time)
        eng_key = "B" if t.signal_type == "COILED_MA200_STAGE2" else "A"
        same_sym_open = any(sym == t.symbol for _, _, sym, _, _ in open_positions)
        eng_open_cnt = sum(1 for _, ek, _, _, _ in open_positions if ek == eng_key)

        if same_sym_open or eng_open_cnt >= slots_per_engine or equity <= 0:
            t.executed = False
            t.pos_leverage = 0.0
            t.acct_ret_pct = 0.0
            t.after_equity_usd = equity
            continue

        executed_count += 1
        lev = min(max_pos_leverage, risk_per_trade_pct / max(t.risk_pct, 0.5))
        acct_ret_pct = lev * t.pnl_pct_net
        pnl_usd = equity * (acct_ret_pct / 100.0)

        t.executed = True
        t.pos_leverage = lev
        t.acct_ret_pct = acct_ret_pct
        total_fee_drag_pct += lev * t.fee_pct
        total_funding_drag_pct += lev * t.funding_pct
        open_positions.append((t.exit_time, eng_key, t.symbol, pnl_usd, t))

    _settle_until(None)

    # 對照組：原版「單倉排他 3x 全倉滾倉」（揭露波動率拖累與大回撤風險）
    naive_cap = init_cap
    naive_peak = init_cap
    naive_mdd = 0.0
    naive_curve = [init_cap]
    last_exit: Optional[datetime] = None
    for t in ordered:
        if last_exit is not None and t.entry_time < last_exit:
            continue
        last_exit = t.exit_time
        naive_cap = max(0.0, naive_cap * (1.0 + naive_full_leverage * t.pnl_pct_net / 100.0))
        naive_peak = max(naive_peak, naive_cap)
        if naive_peak > 0:
            naive_mdd = max(naive_mdd, (naive_peak - naive_cap) / naive_peak * 100.0)
        naive_curve.append(naive_cap)

    return {
        "init_cap": init_cap,
        "risk_per_trade_pct": risk_per_trade_pct,
        "max_pos_leverage": max_pos_leverage,
        "slots_per_engine": slots_per_engine,
        "final_risk_usd": equity,
        "risk_ret_pct": (equity / init_cap - 1.0) * 100.0,
        "risk_mdd_pct": mdd,
        "risk_curve": eq_curve,
        "executed": executed_count,
        "skipped": len(ordered) - executed_count,
        "fee_drag_pct": total_fee_drag_pct,
        "funding_drag_pct": total_funding_drag_pct,
        "naive_lev": naive_full_leverage,
        "final_naive_3x_usd": naive_cap,
        "naive_3x_ret_pct": (naive_cap / init_cap - 1.0) * 100.0,
        "naive_3x_mdd_pct": naive_mdd,
        "naive_curve": naive_curve,
    }


def _engine_sub_stats(sub_trades: Sequence[TradeResult]) -> Dict[str, float]:
    if not sub_trades:
        return {"n": 0, "wins": 0, "wr": 0.0, "total_r": 0.0, "avg_r": 0.0, "total_pct": 0.0, "pf": 0.0, "pf_ex2": 0.0}
    rs = np.array([t.r_net for t in sub_trades], dtype=float)
    pcts = np.array([t.pnl_pct_net for t in sub_trades], dtype=float)
    wins = int(np.sum(rs > 0))
    gw = float(np.sum(rs[rs > 0]))
    gl = abs(float(np.sum(rs[rs < 0])))
    pf = gw / gl if gl > 0 else 99.0
    rs_ex2 = np.sort(rs)[:-2] if len(rs) > 4 else rs
    gw_ex2 = float(np.sum(rs_ex2[rs_ex2 > 0]))
    gl_ex2 = abs(float(np.sum(rs_ex2[rs_ex2 < 0])))
    pf_ex2 = gw_ex2 / gl_ex2 if gl_ex2 > 0 else 99.0
    return {
        "n": len(sub_trades),
        "wins": wins,
        "wr": float(wins / len(sub_trades) * 100.0),
        "total_r": float(np.sum(rs)),
        "avg_r": float(np.mean(rs)),
        "total_pct": float(np.sum(pcts)),
        "pf": pf,
        "pf_ex2": pf_ex2,
    }


def plot_single_trade_card(df: pd.DataFrame, tr: TradeResult, out_path: Path) -> None:
    """繪製包含進場、初始止損、MA99 趨勢移動停利出場點的復盤 K 線圖卡。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "WenQuanYi Micro Hei", "Noto Sans CJK TC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    is_coiled = tr.signal_type == "COILED_MA200_STAGE2"
    left = max(0, tr.entry_idx - 55)
    right = min(len(df) - 1, tr.exit_idx + 18)
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

    entry_vol = float(vols[e_pos]) if 0 <= e_pos < len(vols) else 0.0
    max_vol_all = float(np.nanmax(vols)) if len(vols) else 1.0
    vol_ylim_top = entry_vol * 2.5 if (entry_vol > 0 and max_vol_all > entry_vol * 2.6) else max_vol_all * 1.18
    ax_vol.set_ylim(0, max(vol_ylim_top, 1e-9))

    entry_badge_color = "#6a1b9a" if is_coiled else "#0277bd"
    ax_price.hlines(tr.entry_price, e_pos, x_pos, colors="#1e88e5", linestyles="--", linewidth=1.3, label=f"進場 ({tr.entry_price:.5g})", zorder=5)
    ax_price.hlines(tr.stop_price, max(0, e_pos - 5), min(len(sub) - 1, e_pos + 18), colors="#e53935", linestyles=":", linewidth=1.4, label=f"止損 ({tr.stop_price:.5g})", zorder=5)

    ax_price.scatter([e_pos], [lows[e_pos] * 0.975], marker="^", s=180, color="#fb8c00", edgecolors="white", zorder=8)
    exit_badge_color = "#00897b" if tr.r_net > 0 else "#e53935"
    ax_price.scatter([x_pos], [highs[x_pos] * 1.025], marker="v", s=180, color=exit_badge_color, edgecolors="white", zorder=8)

    y_span = float(np.nanmax(highs) - np.nanmin(lows))
    ax_price.annotate(
        f"★ 二拍進場 {tr.entry_price:.5g}\n({tr.entry_time.strftime('%m-%d %H:%M')}, 24h額 ${tr.prev_24h_quote_vol/1e6:.1f}M)",
        xy=(e_pos, lows[e_pos]),
        xytext=(max(1, e_pos - 15), max(float(np.nanmin(lows)), lows[e_pos] - 0.14 * y_span)),
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.36", "facecolor": entry_badge_color, "alpha": 0.92, "edgecolor": "none"},
        arrowprops={"arrowstyle": "->", "color": entry_badge_color, "lw": 1.3},
        zorder=9,
    )
    ax_price.annotate(
        f"▼ {tr.exit_reason} {tr.exit_price:.5g}\n扣費資淨利: {tr.pnl_pct_net:+.1f}% ({tr.r_net:+.2f}R)",
        xy=(x_pos, highs[x_pos]),
        xytext=(max(2, min(len(sub) - 24, x_pos - 10)), min(float(np.nanmax(highs)) * 0.96, highs[x_pos] + 0.08 * y_span)),
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.36", "facecolor": exit_badge_color, "alpha": 0.92, "edgecolor": "none"},
        arrowprops={"arrowstyle": "->", "color": exit_badge_color, "lw": 1.3},
        zorder=9,
    )

    stype_label = "引擎B·水下穿MA200二拍" if is_coiled else "引擎A·水上高台黏帶二拍"
    ax_price.set_title(
        f"{tr.symbol} ({tr.interval}) [{stype_label}] — 扣費資淨利 {tr.pnl_pct_net:+.1f}% ({tr.r_net:+.2f}R)",
        fontsize=12,
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
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_equity_curve(trades: Sequence[TradeResult], out_path: Path, port_stats: Dict[str, Any]) -> None:
    """繪製「1.5% 固定風險分倉 vs 無腦 3x 全倉滾倉」淨值對照圖與累積 R 倍數曲線。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "WenQuanYi Micro Hei", "Noto Sans CJK TC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if not trades:
        return

    risk_curve = port_stats["risk_curve"]
    naive_curve = port_stats["naive_curve"]
    init_cap = port_stats["init_cap"]

    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(14.8, 5.2),
        facecolor="#ffffff",
        gridspec_kw={"wspace": 0.18},
    )

    ax_left.plot(
        np.linspace(0, 100, len(risk_curve)),
        risk_curve,
        color="#00897b",
        linewidth=2.4,
        label=(
            f"固定風險分倉 ({port_stats['risk_per_trade_pct']:.1f}%R/筆, 最多{port_stats['slots_per_engine']*2}倉): "
            f"${port_stats['final_risk_usd']:,.0f} ({port_stats['risk_ret_pct']:+.1f}%, MDD -{port_stats['risk_mdd_pct']:.1f}%)"
        ),
    )
    ax_left.plot(
        np.linspace(0, 100, len(naive_curve)),
        naive_curve,
        color="#e53935",
        linewidth=1.7,
        linestyle="--",
        label=(
            f"對照：原版全倉 {port_stats['naive_lev']:g}x 滾倉: "
            f"${port_stats['final_naive_3x_usd']:,.0f} ({port_stats['naive_3x_ret_pct']:+.1f}%, MDD -{port_stats['naive_3x_mdd_pct']:.1f}%)"
        ),
    )
    ax_left.axhline(init_cap, color="#263238", linewidth=0.9, linestyle=":")
    ax_left.set_title("倉位風控決定生死：1.5% 固定風險分倉 vs 全倉 3x 滾倉 (USD)", fontsize=11.5, fontweight="bold")
    ax_left.set_ylabel("帳戶權益 (USD)", fontsize=10)
    ax_left.set_xlabel("回測時間進度 (%)", fontsize=10)
    ax_left.grid(True, linestyle="--", alpha=0.3)
    ax_left.legend(loc="upper left", fontsize=8.8, framealpha=0.92)

    eq_r = np.r_[0.0, np.cumsum([t.r_net for t in trades])]
    ax_right.plot(
        range(len(eq_r)),
        eq_r,
        color="#1e88e5",
        linewidth=2.2,
        label=f"全部無重疊訊號純 R 累積 ({len(trades)} 筆: {eq_r[-1]:+.1f}R)",
    )
    ax_right.fill_between(range(len(eq_r)), 0, eq_r, color="#e3f2fd", alpha=0.55)
    ax_right.axhline(0, color="#263238", linewidth=0.9, linestyle=":")
    ax_right.set_title("雙引擎全部合格訊號累積盈虧比曲線 (已扣 0.24% 成本 + 資費)", fontsize=11.5, fontweight="bold")
    ax_right.set_ylabel("累積盈虧比 (R)", fontsize=10)
    ax_right.set_xlabel("訊號推進筆數", fontsize=10)
    ax_right.grid(True, linestyle="--", alpha=0.3)
    ax_right.legend(loc="upper left", fontsize=9.0, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def generate_html_report(
    trades: Sequence[TradeResult],
    featured_trades: Sequence[TradeResult],
    out_html: Path,
    interval: str,
    days: int,
    n_symbols: int,
    port_stats: Dict[str, Any],
    split_cutoff: Optional[datetime] = None,
) -> None:
    """產出誠實揭露交易摩擦、風控對比與樣本內外拆解的 HTML 戰績看板。"""
    st_all = _engine_sub_stats(trades)
    eng_a = [t for t in trades if t.signal_type != "COILED_MA200_STAGE2"]
    eng_b = [t for t in trades if t.signal_type == "COILED_MA200_STAGE2"]
    st_a = _engine_sub_stats(eng_a)
    st_b = _engine_sub_stats(eng_b)

    oos_section_html = ""
    if split_cutoff is not None:
        early = [t for t in trades if t.entry_time < split_cutoff]
        recent = [t for t in trades if t.entry_time >= split_cutoff]
        if early and recent:
            st_e = _engine_sub_stats(early)
            st_r = _engine_sub_stats(recent)
            oos_section_html = f"""
  <h2>🔬 跨期穩健度檢驗（前段歷史區間 vs 近 90 天區間，零參數微調）</h2>
  <table>
    <thead>
      <tr>
        <th>檢驗區間</th><th>訊號筆數</th><th>勝率</th><th>累積盈虧比 (R)</th><th>單筆期望值</th><th>獲利因子 (PF)</th><th>扣除前2大極端值後 PF</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>前段樣本區間（~ {split_cutoff.strftime('%Y-%m-%d')}）</b></td>
        <td><b>{st_e['n']} 筆</b> ({st_e['wins']}勝)</td>
        <td>{st_e['wr']:.1f}%</td>
        <td class="{'pos' if st_e['total_r']>=0 else 'neg'}"><b>{st_e['total_r']:+.1f}R</b></td>
        <td class="{'pos' if st_e['avg_r']>=0 else 'neg'}"><b>{st_e['avg_r']:+.2f}R</b></td>
        <td><b>{st_e['pf']:.2f}</b></td>
        <td><b>{st_e['pf_ex2']:.2f}</b></td>
      </tr>
      <tr>
        <td><b>近期驗證區間（{split_cutoff.strftime('%Y-%m-%d')} ~ 今）</b></td>
        <td><b>{st_r['n']} 筆</b> ({st_r['wins']}勝)</td>
        <td>{st_r['wr']:.1f}%</td>
        <td class="{'pos' if st_r['total_r']>=0 else 'neg'}"><b>{st_r['total_r']:+.1f}R</b></td>
        <td class="{'pos' if st_r['avg_r']>=0 else 'neg'}"><b>{st_r['avg_r']:+.2f}R</b></td>
        <td><b>{st_r['pf']:.2f}</b></td>
        <td><b>{st_r['pf_ex2']:.2f}</b></td>
      </tr>
    </tbody>
  </table>
"""

    win_cards_html = []
    loss_cards_html = []
    for t in featured_trades:
        if not t.chart_file:
            continue
        is_win = t.r_net > 0
        badge_bg = "#00c853" if is_win else "#ff5252"
        if t.signal_type == "COILED_MA200_STAGE2":
            stype = "🐂 引擎B·水下穿MA200二拍"
            tag_style = "background:#6a1b9a;color:#fff;"
        else:
            stype = "★ 引擎A·水上高台黏帶二拍"
            tag_style = "background:#0277bd;color:#fff;"
        exec_tag = (
            f"實執倉位 {t.pos_leverage:.2f}x: 帳戶 {t.acct_ret_pct:+.2f}% → 權益 ${t.after_equity_usd:,.0f}"
            if t.executed
            else "滿倉風控跳過"
        )
        card_block = f"""
            <div class="card">
              <div class="card-header">
                <div>
                  <span class="sym">{escape(t.symbol)}</span>
                  <span class="tag" style="{tag_style}">{stype}</span>
                  <span class="time">{t.entry_time.strftime('%Y-%m-%d %H:%M')} TPE</span>
                </div>
                <div class="pnl-badge" style="background:{badge_bg}">
                  1x淨利: {t.pnl_pct_net:+.1f}% ({t.r_net:+.2f}R)
                </div>
              </div>
              <div class="card-meta">
                進場: <b>{t.entry_price:.6g}</b> ｜ 止損: <b>{t.stop_price:.6g}</b> (風險 {t.risk_pct:.1f}%) ｜
                出場: <b>{t.exit_price:.6g}</b> ({escape(t.exit_reason)}) ｜ <b>{exec_tag}</b>
              </div>
              <div class="card-note">{escape(t.note)}</div>
              <a href="cards/{escape(t.chart_file)}" target="_blank">
                <img src="cards/{escape(t.chart_file)}" loading="lazy" alt="{escape(t.symbol)}"/>
              </a>
            </div>
        """
        if is_win:
            win_cards_html.append(card_block)
        else:
            loss_cards_html.append(card_block)

    rows_html = []
    for idx, t in enumerate(reversed(trades), 1):
        color_cls = "pos" if t.r_net > 0 else "neg"
        stype = "🐂 引擎B·水下二拍" if t.signal_type == "COILED_MA200_STAGE2" else "★ 引擎A·水上二拍"
        acct_cell = f"{t.acct_ret_pct:+.2f}% ({t.pos_leverage:.2f}x)" if t.executed else "滿倉略過"
        eq_cell = f"${t.after_equity_usd:,.2f}" if t.executed else f"(${t.after_equity_usd:,.2f})"
        rows_html.append(
            f"""
            <tr>
              <td>{idx}</td>
              <td>{t.entry_time.strftime('%Y-%m-%d %H:%M')}</td>
              <td><b>{escape(t.symbol)}</b></td>
              <td>{stype}</td>
              <td>${t.prev_24h_quote_vol/1e6:.1f}M</td>
              <td>{t.entry_price:.6g}</td>
              <td>{t.stop_price:.6g} ({t.risk_pct:.1f}%)</td>
              <td>{t.exit_price:.6g}</td>
              <td>{escape(t.exit_reason)}</td>
              <td class="{color_cls}"><b>{t.pnl_pct_net:+.2f}%</b></td>
              <td class="{color_cls}"><b>{acct_cell}</b></td>
              <td><b>{eq_cell}</b></td>
              <td class="{color_cls}"><b>{t.r_net:+.2f}R</b></td>
            </tr>
            """
        )

    naive_cls = "green" if port_stats["naive_3x_ret_pct"] >= 0 else ""
    naive_style = "" if port_stats["naive_3x_ret_pct"] >= 0 else 'style="color:#f87171"'

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>幣安永續「結構雙引擎二拍突破（去過擬合・零前視偏差版）」{days}天實盤級回測報告 ({interval})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Microsoft JhengHei", sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
  .container {{ max-width: 1440px; margin: 0 auto; }}
  h1 {{ font-size: 25px; margin-bottom: 6px; color: #f8fafc; }}
  .sub {{ color: #94a3b8; font-size: 13.5px; line-height: 1.6; margin-bottom: 22px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 14px; margin-bottom: 24px; }}
  .kpi {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; }}
  .kpi .label {{ font-size: 13px; color: #94a3b8; }}
  .kpi .val {{ font-size: 24px; font-weight: 800; margin-top: 6px; color: #38bdf8; }}
  .kpi .val.green {{ color: #4ade80; }}
  .kpi .val.amber {{ color: #fbbf24; }}
  .eq-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 28px; text-align: center; }}
  .eq-box img {{ max-width: 100%; border-radius: 8px; }}
  h2 {{ font-size: 19px; margin: 28px 0 14px; color: #f1f5f9; border-left: 4px solid #38bdf8; padding-left: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 14px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .sym {{ font-size: 18px; font-weight: 800; color: #f8fafc; margin-right: 8px; }}
  .tag {{ background: #334155; color: #e2e8f0; padding: 3px 8px; border-radius: 6px; font-size: 12px; margin-right: 8px; font-weight: 600; }}
  .time {{ color: #94a3b8; font-size: 13px; }}
  .pnl-badge {{ color: #fff; font-weight: 800; padding: 4px 10px; border-radius: 8px; font-size: 13.5px; }}
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
  <h1>🛡️ 幣安永續合約「結構雙引擎二拍突破（去過擬合・零前視偏差修正版）」過去 {days} 天回測報告</h1>
  <div class="sub">
    K線週期：<b>{escape(interval)}（僅已收盤K線）</b> ｜ 掃描標的池：<b>{n_symbols} 檔 USDT 永續合約（訊號當下強制要求前 24h 成交額 &ge; $5M，絕不硬塞事後妖幣）</b><br/>
    嚴格實盤假設：<b>已扣 0.24% 來回手續費與滑價 ＋ 持倉每 8h 資金費率 0.01% ＋ 止損開盤跳空滑價 ＋ 收盤站上 +1.5R 次根才移保本（杜絕同根幽靈保本）</b>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">初始本金 / 單筆風險預算</div><div class="val">${port_stats['init_cap']:,.0f} ({port_stats['risk_per_trade_pct']:.1f}%R)</div></div>
    <div class="kpi"><div class="label">1.5%R 分倉最終權益</div><div class="val green">${port_stats['final_risk_usd']:,.2f} ({port_stats['risk_ret_pct']:+.1f}%)</div></div>
    <div class="kpi"><div class="label">1.5%R 分倉最大回撤</div><div class="val amber">-{port_stats['risk_mdd_pct']:.1f}%</div></div>
    <div class="kpi"><div class="label">對照：全倉 3x 滾倉結局</div><div class="val {naive_cls}" {naive_style}>${port_stats['final_naive_3x_usd']:,.0f} ({port_stats['naive_3x_ret_pct']:+.1f}%)</div></div>
    <div class="kpi"><div class="label">總訊號勝率 ({st_all['n']} 筆)</div><div class="val">{st_all['wr']:.1f}%</div></div>
    <div class="kpi"><div class="label">總獲利因子 PF (扣前2極端值)</div><div class="val amber">{st_all['pf']:.2f} ({st_all['pf_ex2']:.2f})</div></div>
    <div class="kpi"><div class="label">累計手續費滑價 / 資費拖累</div><div class="val" style="color:#f87171">-{port_stats['fee_drag_pct']:.1f}% / -{port_stats['funding_drag_pct']:.2f}%</div></div>
  </div>

  <h2>⚡ 雙引擎結構分項表現（0% 訊號重疊・全整數結構門檻・無小數點湊參數）</h2>
  <table>
    <thead>
      <tr>
        <th>子引擎名稱</th><th>核心結構條件與統一出場規則</th><th>訊號數</th><th>勝率</th><th>1x 累計淨利</th><th>累積盈虧比 (R)</th><th>PF (去極端值PF)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>★ 引擎 A（水上高台黏帶二拍）</b></td>
        <td>MA200上揚且穩守&ge;80根 + 7線黏帶&le;2% + 箱體&le;5% + 首拍2~8x破箱頂 + 二拍放量確認 → 沿 MA99 移動停利</td>
        <td><b>{st_a['n']} 筆</b> ({st_a['wins']}勝)</td>
        <td><b>{st_a['wr']:.1f}%</b></td>
        <td class="{'pos' if st_a['total_pct']>=0 else 'neg'}"><b>{st_a['total_pct']:+.1f}%</b></td>
        <td class="{'pos' if st_a['total_r']>=0 else 'neg'}"><b>{st_a['total_r']:+.1f}R</b></td>
        <td><b>{st_a['pf']:.2f} ({st_a['pf_ex2']:.2f})</b></td>
      </tr>
      <tr>
        <td><b>🐂 引擎 B（水下築底穿MA200二拍）</b></td>
        <td>MA200水下築底&ge;80根 + 7線黏帶&le;2% + 首拍2~8x穿MA200 + 二拍穩守MA200放量確認 → 沿 MA99 移動停利</td>
        <td><b>{st_b['n']} 筆</b> ({st_b['wins']}勝)</td>
        <td><b>{st_b['wr']:.1f}%</b></td>
        <td class="{'pos' if st_b['total_pct']>=0 else 'neg'}"><b>{st_b['total_pct']:+.1f}%</b></td>
        <td class="{'pos' if st_b['total_r']>=0 else 'neg'}"><b>{st_b['total_r']:+.1f}R</b></td>
        <td><b>{st_b['pf']:.2f} ({st_b['pf_ex2']:.2f})</b></td>
      </tr>
      <tr style="background:#0f172a;">
        <td><b>🛡️ 雙引擎合體總計（A/B 各最多 {port_stats['slots_per_engine']} 倉）</b></td>
        <td>實際執行 {port_stats['executed']} 筆 / 滿倉風控略過 {port_stats['skipped']} 筆（對照全倉 3x 滾倉最大回撤高達 -{port_stats['naive_3x_mdd_pct']:.1f}%）</td>
        <td><b>{st_all['n']} 筆</b> ({st_all['wins']}勝)</td>
        <td><b>{st_all['wr']:.1f}%</b></td>
        <td class="{'pos' if st_all['total_pct']>=0 else 'neg'}"><b>{st_all['total_pct']:+.1f}%</b></td>
        <td class="{'pos' if st_all['total_r']>=0 else 'neg'}"><b>{st_all['total_r']:+.1f}R</b></td>
        <td><b>{st_all['pf']:.2f} ({st_all['pf_ex2']:.2f})</b></td>
      </tr>
    </tbody>
  </table>

  {oos_section_html}

  <div class="eq-box">
    <img src="equity_curve.png" alt="Equity Curve"/>
  </div>

  <h2>📈 代表性獲利交易復盤圖卡（前 {len(win_cards_html)} 筆）</h2>
  <div class="grid">
    {''.join(win_cards_html)}
  </div>

  <h2>📉 代表性虧損交易復盤圖卡（前 {len(loss_cards_html)} 筆）</h2>
  <div class="grid">
    {''.join(loss_cards_html)}
  </div>

  <h2>📋 過去 {days} 天完整合格訊號明細表（共 {st_all['n']} 筆，依時間由新到舊）</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>進場時間 (TPE)</th><th>幣種</th><th>引擎類型</th><th>當下24h額</th>
        <th>進場價</th><th>止損價 (風險)</th><th>出場價</th><th>出場原因</th>
        <th>1x扣費資淨利</th><th>風控倉位損益</th><th>平倉後帳戶權益</th><th>盈虧比 (R)</th>
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
    write_view_html(out_html)


def _git_branch(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
        return out or "main"
    except Exception:
        return "main"


def write_view_html(src_html: Path, repo_slug: str = "yubogoodman-droid/binance-dual-engine-v8") -> Path:
    """產出供 htmlpreview.github.io 免部署即時開啟的 view.html（圖片轉為 raw.githubusercontent 絕對網址）。"""
    repo_root = Path(__file__).resolve().parent
    src_resolved = src_html.resolve()
    try:
        rel_dir = src_resolved.parent.relative_to(repo_root).as_posix()
    except ValueError:
        rel_dir = src_html.parent.as_posix()
    branch = _git_branch(repo_root)
    raw_base = f"https://raw.githubusercontent.com/{repo_slug}/{branch}/{rel_dir}/"
    content = (
        src_resolved.read_text(encoding="utf-8")
        .replace('src="equity_curve.png"', f'src="{raw_base}equity_curve.png"')
        .replace('src="cards/', f'src="{raw_base}cards/')
        .replace('href="cards/', f'href="{raw_base}cards/')
    )
    view_path = src_resolved.with_name("view.html")
    view_path.write_text(content, encoding="utf-8")
    return view_path


def summarize_trades(trades: Sequence[TradeResult], label: str) -> None:
    if not trades:
        print(f"[{label}] 無交易紀錄")
        return
    st = _engine_sub_stats(trades)
    print(
        f"  {label:<34} | 訊號數: {st['n']:>4} | 勝率: {st['wr']:>5.1f}% | "
        f"總獲利: {st['total_r']:>+7.1f}R | 期望值: {st['avg_r']:>+5.2f}R/筆 | "
        f"PF: {st['pf']:>5.2f} (去前2極端值 PF: {st['pf_ex2']:>4.2f})"
    )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="幣安永續合約結構雙引擎無前視偏差回測引擎")
    parser.add_argument("--interval", default="15m", help="K線週期 (預設 15m)")
    parser.add_argument("--days", type=int, default=90, help="回測天數 (例如 90 或 365)")
    parser.add_argument("--top", type=int, default=180, help="掃描成交額前 N 大 USDT 永續合約")
    parser.add_argument("--min-qv", type=float, default=scanner.DEFAULT_MIN_QUOTE_VOL, help="訊號當下最低前 24h 成交額 (預設 5,000,000)")
    parser.add_argument("--capital", type=float, default=1000.0, help="初始本金 (USD)")
    parser.add_argument("--risk-pct", type=float, default=1.5, help="單筆固定帳戶風險百分比 (預設 1.5%%)")
    parser.add_argument("--fee-pct", type=float, default=FEE_ROUNDTRIP_PCT, help="來回手續費+滑價百分比 (預設 0.24%%)")
    parser.add_argument("--max-cards", type=int, default=24, help="最多輸出幾張代表性 K 線復盤圖卡 (預設 24)")
    parser.add_argument("--cache-dir", type=str, default="", help="選填：本機 .npz K 線快取資料夾路徑")
    parser.add_argument("--end-time", type=str, default="", help="選填：固定回測結束時間 (YYYY-MM-DD HH:MM, TPE)")
    parser.add_argument("--out-dir", default="breakout_output/backtest_90d", help="報表輸出目錄")
    parser.add_argument("--html", action="store_true", help="產生 HTML 戰績看板（預設自動開啟）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cards_dir = out_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for old_png in cards_dir.glob("*.png"):
        old_png.unlink()

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    npz_files: Dict[str, Path] = {}
    if cache_dir and cache_dir.is_dir():
        for p in sorted(cache_dir.glob("*.npz")):
            sym = p.stem.replace("_730d", "")
            if sym not in scanner.EXCLUDED_SYMBOLS:
                npz_files[sym] = p

    if npz_files:
        all_universe_syms = sorted(npz_files.keys())
    else:
        all_universe_syms = scanner.fetch_top_usdt_perps(top_n=args.top, min_quote_vol=args.min_qv)
        if "BTCUSDT" not in all_universe_syms:
            all_universe_syms = ["BTCUSDT"] + list(all_universe_syms)

    end_dt = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M").replace(tzinfo=TPE) if args.end_time else datetime.now(TPE)
    cutoff = end_dt - timedelta(days=args.days)
    crop_dt = cutoff - timedelta(days=15)

    print(
        f"正在對 {len(all_universe_syms)} 檔 USDT 永續合約 ({cutoff.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}, {args.interval}) "
        f"執行零前視偏差結構雙引擎回測..."
    )

    all_dfs: Dict[str, pd.DataFrame] = {}
    if npz_files:
        for sym in all_universe_syms:
            df = load_cached_npz(npz_files[sym], interval=args.interval)
            df = df[(df.index >= crop_dt) & (df.index <= end_dt)]
            if len(df) > 235:
                all_dfs[sym] = df
    else:
        def _fetch(sym: str) -> tuple[str, pd.DataFrame]:
            df = fetch_klines_paginated(sym, interval=args.interval, days=args.days)
            if not df.empty:
                df = df[(df.index >= crop_dt) & (df.index <= end_dt)]
            return sym, df

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_fetch, sym): sym for sym in all_universe_syms}
            for fut in as_completed(futs):
                sym, df = fut.result()
                if len(df) > 235:
                    all_dfs[sym] = df

    btc_safe_map = scanner.build_btc_regime_map(all_dfs["BTCUSDT"]) if "BTCUSDT" in all_dfs else {}

    raw_signals: List[scanner.BreakoutSignal] = []
    for sym, df in all_dfs.items():
        sigs = (
            scanner.detect_signals(df, sym, args.interval, min_quote_vol_24h=args.min_qv)
            + scanner.detect_coiled_ma200_signals(df, sym, args.interval, min_quote_vol_24h=args.min_qv)
        )
        raw_signals.extend(s for s in sigs if s.timestamp >= cutoff)

    ranked_signals = scanner.filter_and_rank_signals(raw_signals, btc_safe_map=btc_safe_map, max_same_bar_signals=2)

    all_trades: List[TradeResult] = []
    for sig in ranked_signals:
        tr = simulate_structural_trade(all_dfs[sig.symbol], sig, fee_roundtrip_pct=args.fee_pct)
        if tr is not None:
            all_trades.append(tr)

    port_stats = compute_portfolio_simulation(
        all_trades,
        init_cap=args.capital,
        risk_per_trade_pct=args.risk_pct,
    )

    eng_a_trades = [t for t in all_trades if t.signal_type != "COILED_MA200_STAGE2"]
    eng_b_trades = [t for t in all_trades if t.signal_type == "COILED_MA200_STAGE2"]

    print("\n" + "=" * 120)
    print(
        f"【結構雙引擎修正版 — {args.days} 天回測總結】標的池: {len(all_dfs)} 檔 | "
        f"引擎A: {len(eng_a_trades)} 筆 + 引擎B: {len(eng_b_trades)} 筆 = 共 {len(all_trades)} 筆 (執行 {port_stats['executed']} 筆)"
    )
    print("-" * 120)
    summarize_trades(eng_a_trades, "★ 引擎 A (水上高台黏帶二拍)")
    summarize_trades(eng_b_trades, "🐂 引擎 B (水下築底穿MA200二拍)")
    summarize_trades(all_trades, "🛡️ 雙引擎全部合格訊號合計")
    print(
        f"  倉位風控對比 | 1.5%R 分倉最終: ${port_stats['final_risk_usd']:,.2f} ({port_stats['risk_ret_pct']:+.1f}%, MDD -{port_stats['risk_mdd_pct']:.1f}%) "
        f"vs 原版全倉 3x 滾倉: ${port_stats['final_naive_3x_usd']:,.2f} ({port_stats['naive_3x_ret_pct']:+.1f}%, MDD -{port_stats['naive_3x_mdd_pct']:.1f}%)"
    )
    print("=" * 120)

    plot_equity_curve(all_trades, out_dir / "equity_curve.png", port_stats)

    half_cards = max(4, args.max_cards // 2)
    top_winners = sorted([t for t in all_trades if t.r_net > 0], key=lambda t: t.r_net, reverse=True)[:half_cards]
    top_losers = sorted([t for t in all_trades if t.r_net <= 0], key=lambda t: t.r_net)[:half_cards]

    featured: List[TradeResult] = []
    for t in top_winners + top_losers:
        key = f"{t.symbol}_{t.entry_time.strftime('%Y%m%d_%H%M')}"
        t.chart_file = f"{key}.png"
        plot_single_trade_card(all_dfs[t.symbol], t, cards_dir / t.chart_file)
        featured.append(t)

    split_cutoff = end_dt - timedelta(days=90) if args.days >= 180 else None
    html_path = out_dir / "index.html"
    generate_html_report(
        all_trades,
        featured,
        html_path,
        interval=args.interval,
        days=args.days,
        n_symbols=len(all_dfs),
        port_stats=port_stats,
        split_cutoff=split_cutoff,
    )
    print(f"\n[HTML 戰績看板已產生] {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
