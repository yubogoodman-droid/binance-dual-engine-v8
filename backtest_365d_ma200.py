"""MA200 均線粘合起漲策略 — 365 天全市場回測與優化前後對比報表。"""

from __future__ import annotations

import argparse
import copy
import html
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_utils as bt_utils
import ma200_coil_scanner as scanner


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# 優化前初版 STRICT 規則 (未納入量能加速 vol_acc>=1.15、RSI6 過熱上限 <=88.5、MA99/MA200 走平與 18h 微利釋放)
BASELINE_V1_PARAMS: Dict[str, float] = {
    **scanner.SNIPER_PARAMS,
    "min_abv_vwap": 7,
    "min_vol_acc": 0.0,
    "min_quote_vol": 150_000.0,
    "max_rsi6": 100.0,
    "min_slp99_pct": -99.0,
    "min_slp200_pct": -0.30,
}

SPLIT_CUTOFF = pd.Timestamp("2026-05-15", tz="Asia/Taipei")


@dataclass
class VariantReport:
    """單一策略版本的 365 天完整回測摘要與交易明細。"""

    key: str
    title: str
    signals: List[scanner.MA200CoilSignal]
    exit_map: Dict[Tuple[str, datetime], Dict[str, Any]]
    port_1slot: Dict[str, Any]
    port_2slot: Dict[str, Any]
    stats: Dict[str, Any]
    oos_report: Optional[Dict[str, Any]] = None


def _load_cached_market(cache_dir: Path) -> Tuple[Dict[str, pd.DataFrame], str, str]:
    """讀取本地 pickle 快取並一次性計算全市場技術指標（不按字母截斷）。"""
    raw_map = bt_utils.load_cached_frames([cache_dir], interval="15m", excluded_symbols=scanner.EXCLUDED_SYMBOLS)
    if not raw_map:
        raise FileNotFoundError(f"找不到快取檔案: {cache_dir}")

    dfs: Dict[str, pd.DataFrame] = {}
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None

    for sym, raw in sorted(raw_map.items()):
        if raw.empty:
            continue
        df = scanner.compute_indicators(raw)
        dfs[sym] = df
        t0, t1 = df.index[0], df.index[-1]
        min_ts = t0 if min_ts is None or t0 < min_ts else min_ts
        max_ts = t1 if max_ts is None or t1 > max_ts else max_ts

    start_str = min_ts.strftime("%Y-%m-%d") if min_ts is not None else "N/A"
    end_str = max_ts.strftime("%Y-%m-%d") if max_ts is not None else "N/A"
    return dfs, start_str, end_str


def _calc_split_stats(
    sigs: Sequence[scanner.MA200CoilSignal],
    exit_map: Dict[Tuple[str, datetime], Dict[str, Any]],
    split_cutoff: pd.Timestamp = SPLIT_CUTOFF,
) -> Dict[str, Any]:
    """計算整體與樣本內(IS) / 樣本外(OOS)切片統計。"""
    if not sigs:
        empty = {"n": 0, "wr": 0.0, "avg_pnl": 0.0, "sum_pnl": 0.0, "avg_r": 0.0, "sum_r": 0.0, "pf": 0.0}
        return {**empty, "max_streak": 0, "avg_hold_h": 0.0, "h1": empty, "h2": empty}

    def _agg(subset: Sequence[scanner.MA200CoilSignal]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0, "wr": 0.0, "avg_pnl": 0.0, "sum_pnl": 0.0, "avg_r": 0.0, "sum_r": 0.0, "pf": 0.0}
        pnls = np.array([exit_map[(s.symbol, s.timestamp)]["pnl_pct"] for s in subset], dtype=float)
        rs = np.array([exit_map[(s.symbol, s.timestamp)]["r_mult"] for s in subset], dtype=float)
        gp = float(np.sum(pnls[pnls > 0]))
        gl = float(abs(np.sum(pnls[pnls <= 0])))
        return {
            "n": len(subset),
            "wr": float(np.mean(pnls > 0) * 100.0),
            "avg_pnl": float(np.mean(pnls)),
            "sum_pnl": float(np.sum(pnls)),
            "avg_r": float(np.mean(rs)),
            "sum_r": float(np.sum(rs)),
            "pf": gp / gl if gl > 0 else 99.9,
        }

    overall = _agg(sigs)
    streak = 0
    max_streak = 0
    holds_h: List[float] = []
    for s in sigs:
        ex = exit_map[(s.symbol, s.timestamp)]
        holds_h.append((ex["exit_idx"] - s.bar_idx) * 0.25)
        if ex["pnl_pct"] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    h1_sigs = [s for s in sigs if pd.Timestamp(s.timestamp) < split_cutoff]
    h2_sigs = [s for s in sigs if pd.Timestamp(s.timestamp) >= split_cutoff]
    return {
        **overall,
        "max_streak": max_streak,
        "avg_hold_h": float(np.mean(holds_h)),
        "h1": _agg(h1_sigs),
        "h2": _agg(h2_sigs),
    }


def _run_variant(
    dfs: Dict[str, pd.DataFrame],
    key: str,
    title: str,
    pit_index: bt_utils.PointInTimeVolumeIndex,
    funding_store: bt_utils.FundingRateStore,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    preset_override: Optional[Dict[str, float]] = None,
    preset_name: str = "sniper",
    max_hold_bars: int = 288,
    arm_r: float = 1.2,
    scratch_bars: int = 0,
    fee_pct: float = 0.12,
    min_vol_usd: float = 1_000_000.0,
    top_n: int = 0,
    risk_pct: float = 2.0,
    oos_date: str = "",
    oos_ratio: float = 0.50,
    wf_folds: int = 4,
    apply_liquidity_cap: bool = True,
    max_bar_vol_share: float = 0.02,
) -> VariantReport:
    """執行單一策略參數設定的全市場無偏誤掃描、資金費率扣除與誠實資金模擬。"""
    orig = copy.deepcopy(scanner.PRESETS[preset_name])
    if preset_override is not None:
        scanner.PRESETS[preset_name].update(preset_override)

    sigs: List[scanner.MA200CoilSignal] = []
    exit_map: Dict[Tuple[str, datetime], Dict[str, Any]] = {}
    try:
        for sym, df in sorted(dfs.items()):
            for s in scanner.detect_ma200_coil_breakouts(df, sym, interval="15m", preset=preset_name):
                ok_vol, qv24, _ = pit_index.check_eligibility(
                    sym, s.timestamp, min_quote_vol=min_vol_usd, top_n=top_n
                )
                if not ok_vol:
                    continue
                sim = scanner.simulate_ma99_trade(
                    df,
                    s,
                    max_hold_bars=max_hold_bars,
                    arm_r=arm_r,
                    scratch_bars=scratch_bars,
                    fee_roundtrip_pct=fee_pct,
                    funding_store=funding_store,
                    trailing_24h_qv_usd=qv24,
                )
                if sim is not None:
                    sigs.append(s)
                    exit_map[(sym, s.timestamp)] = sim
    finally:
        scanner.PRESETS[preset_name].update(orig)

    sigs.sort(key=lambda s: s.timestamp)
    port_1 = scanner.simulate_portfolio(
        sigs,
        exit_map,
        risk_pct_per_trade=risk_pct,
        max_slots=1,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
    )
    port_2 = scanner.simulate_portfolio(
        sigs,
        exit_map,
        risk_pct_per_trade=risk_pct,
        max_slots=2,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
    )

    inputs = [
        bt_utils.PortfolioTradeInput(
            symbol=s.symbol,
            entry_time=s.timestamp,
            exit_time=exit_map[(s.symbol, s.timestamp)]["exit_time"],
            entry_price=s.entry_price,
            stop_price=s.stop_price,
            exit_price=exit_map[(s.symbol, s.timestamp)]["exit_price"],
            risk_pct=s.risk_pct,
            net_pnl_pct=exit_map[(s.symbol, s.timestamp)]["pnl_pct"],
            r_mult=exit_map[(s.symbol, s.timestamp)]["r_mult"],
            worst_low_price=exit_map[(s.symbol, s.timestamp)]["worst_low_price"],
            bar_quote_vol_usd=exit_map[(s.symbol, s.timestamp)]["bar_quote_vol_usd"],
            trailing_24h_qv_usd=exit_map[(s.symbol, s.timestamp)]["trailing_24h_qv_usd"],
            funding_cost_pct=exit_map[(s.symbol, s.timestamp)]["funding_cost_pct"],
            priority=float(s.score),
        )
        for s in sigs
    ]
    oos_report = bt_utils.compute_oos_and_walkforward(
        inputs,
        window_start=window_start.to_pydatetime(),
        window_end=window_end.to_pydatetime(),
        oos_split_date=oos_date,
        oos_ratio=oos_ratio,
        wf_folds=wf_folds,
        risk_pct_per_trade=risk_pct,
        max_positions=2,
    )
    stats = _calc_split_stats(sigs, exit_map, split_cutoff=oos_report["split_ts"])
    return VariantReport(
        key=key,
        title=title,
        signals=sigs,
        exit_map=exit_map,
        port_1slot=port_1,
        port_2slot=port_2,
        stats=stats,
        oos_report=oos_report,
    )


def _date_curve_points(
    port: Dict[str, Any],
    start_ts: pd.Timestamp,
) -> Tuple[List[pd.Timestamp], List[float], List[float]]:
    """取出按平倉時間軸排列的 2% 以損定倉淨值序列與 3x 理論對照序列。"""
    if port.get("curve_risk2_times") and len(port["curve_risk2_times"]) == len(port["curve_risk2"]):
        xs = [pd.Timestamp(t) for t in port["curve_risk2_times"]]
    else:
        xs = [start_ts]
    ys_risk2 = [float(v) for v in port["curve_risk2"]]
    ys_comp = [float(v) for v in port["curve_comp"]]
    return xs, ys_comp, ys_risk2


def plot_equity_comparison(
    variants: Dict[str, VariantReport],
    n_syms: int,
    start_str: str,
    end_str: str,
    out_path: Path,
) -> None:
    """繪製優化前後 365 天時間軸誠實資金曲線對比圖 + Sniper 全訊號單筆淨利柱狀圖。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    base = variants["baseline"]
    sniper = variants["sniper"]
    bal = variants["balanced"]
    start_ts = pd.Timestamp(start_str, tz="Asia/Taipei")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.2, 8.8), gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.28})
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_color("#334155")

    rpct = sniper.port_2slot.get("risk_pct_per_trade", 2.0)
    curves = [
        (
            sniper.port_2slot,
            "risk2",
            "#4ade80",
            "-",
            2.6,
            f"★ 主指標 Sniper 雙槽 {rpct:g}% 以損定倉 (${sniper.port_2slot['eq_risk2']:,.0f}, {sniper.port_2slot['risk2_ret_pct']:+.1f}%, MDD {sniper.port_2slot['mdd_risk2']:.1f}%)",
        ),
        (
            sniper.port_2slot,
            "spot1x",
            "#22d3ee",
            "--",
            2.0,
            f"Sniper 雙槽 1x 真實複利 (${sniper.port_2slot['eq_spot_1x']:,.0f}, {sniper.port_2slot['spot_1x_ret_pct']:+.1f}%, MDD {sniper.port_2slot['mdd_spot_1x']:.1f}%)",
        ),
        (
            bal.port_2slot,
            "risk2",
            "#a78bfa",
            ":",
            2.0,
            f"Balanced 雙槽 {rpct:g}% 以損定倉 (${bal.port_2slot['eq_risk2']:,.0f}, {bal.port_2slot['risk2_ret_pct']:+.1f}%, MDD {bal.port_2slot['mdd_risk2']:.1f}%)",
        ),
        (
            base.port_1slot,
            "risk2",
            "#f87171",
            "-.",
            1.8,
            f"Baseline v1 單倉 {rpct:g}% 以損定倉 (${base.port_1slot['eq_risk2']:,.0f}, {base.port_1slot['risk2_ret_pct']:+.1f}%, MDD {base.port_1slot['mdd_risk2']:.1f}%)",
        ),
        (
            sniper.port_2slot,
            "comp3x",
            "#fbbf24",
            ":",
            1.6,
            f"Sniper 雙槽 3x [含強平/流動性上限·理論對照] (${sniper.port_2slot['eq_comp']:,.0f}, MDD {sniper.port_2slot['mdd_comp']:.1f}%)",
        ),
    ]

    for port, kind, color, ls, lw, label in curves:
        if kind == "risk2":
            ts_seq = [pd.Timestamp(t) for t in port.get("curve_risk2_times", [])] or [start_ts]
            ys_seq = [float(v) for v in port["curve_risk2"]]
        elif kind == "spot1x":
            ts_seq = [pd.Timestamp(t) for t in port.get("curve_spot_1x_times", [])] or [start_ts]
            ys_seq = [float(v) for v in port["curve_spot_1x"]]
        else:
            ts_seq = [pd.Timestamp(t) for t in port.get("curve_comp_times", [])] or [start_ts]
            ys_seq = [float(v) for v in port["curve_comp"]]
        if len(ts_seq) == len(ys_seq) and ys_seq:
            ax1.step(ts_seq, ys_seq, where="post", color=color, linestyle=ls, linewidth=lw, label=label)
            ax1.scatter(ts_seq[-1:], ys_seq[-1:], color=color, s=36, zorder=5)

    ax1.set_yscale("log")
    ax1.set_ylabel("帳戶淨值 (USDT, 對數刻度)", color="#cbd5e1", fontsize=10.5)
    ax1.set_title(
        f"MA200 均線粘合起漲策略 — 365 天全市場誠實資金曲線 ({start_str} ~ {end_str} | {n_syms} 檔合約 | 已扣費滑 + 資金費率)",
        fontsize=12.5,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.22, color="#475569")
    ax1.legend(loc="upper left", fontsize=8.8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    # 下半部：優化後 Sniper 全訊號 1x 淨利柱狀圖
    sigs = sniper.signals
    x_tr = np.arange(len(sigs))
    pnls = [sniper.exit_map[(s.symbol, s.timestamp)]["pnl_pct"] for s in sigs]
    cols = ["#0ecb81" if p > 0 else "#f6465d" for p in pnls]
    bars = ax2.bar(x_tr, pnls, color=cols, width=0.64, alpha=0.92)
    ax2.axhline(0, color="#64748b", linewidth=1.0)
    for bar, p in zip(bars, pnls):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            p + (1.4 if p >= 0 else -1.8),
            f"{p:+.1f}%",
            ha="center",
            va="bottom" if p >= 0 else "top",
            fontsize=7.6,
            color="#e2e8f0",
            fontweight="bold",
        )
    ax2.set_xticks(x_tr)
    ax2.set_xticklabels(
        [f"{s.symbol.replace('USDT', '')}\n{s.timestamp.strftime('%m-%d')}" for s in sigs],
        fontsize=7.3,
        color="#cbd5e1",
    )
    ax2.set_ylabel("單筆 1x 淨利 (%)", color="#cbd5e1")
    ax2.set_title(
        f"Sniper 狙擊模式全體 {len(sigs)} 筆訊號單筆表現 (勝率 {sniper.stats['wr']:.1f}% / PF {sniper.stats['pf']:.2f} / 總R {sniper.stats['sum_r']:+.1f}R)",
        fontsize=10.5,
        color="#e2e8f0",
        pad=6,
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.22, color="#475569")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_trade_cards(
    dfs: Dict[str, pd.DataFrame],
    report: VariantReport,
    cards_dir: Path,
) -> Dict[Tuple[str, datetime], str]:
    """替 Sniper 全體訊號輸出獨立 K 線復盤圖卡並回傳相對路徑。"""
    cards_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: Dict[Tuple[str, datetime], str] = {}
    for sig in report.signals:
        df = dfs[sig.symbol]
        ex = report.exit_map[(sig.symbol, sig.timestamp)]
        fname = f"{sig.symbol}_{sig.timestamp.strftime('%Y%m%d_%H%M')}.png"
        out_path = cards_dir / fname
        scanner.plot_signal_chart(
            df,
            [sig],
            sig.symbol,
            "15m",
            out_path,
            lookback_bars=150,
            exit_map={sig.timestamp: ex},
        )
        rel_paths[(sig.symbol, sig.timestamp)] = f"cards/{fname}"
    return rel_paths


def write_html_report(
    out_dir: Path,
    variants: Dict[str, VariantReport],
    card_paths: Dict[Tuple[str, datetime], str],
    n_syms: int,
    start_str: str,
    end_str: str,
    fee_pct: float,
) -> Path:
    """輸出深色視覺化 HTML 回測儀表板。"""
    sniper = variants["sniper"]
    base = variants["baseline"]
    bal = variants["balanced"]
    st_s = sniper.stats
    p1 = sniper.port_1slot
    p2 = sniper.port_2slot

    exec2_keys = {(s.symbol, s.timestamp) for s, _, _, _ in p2["executed_items"]}
    exec1_keys = {(s.symbol, s.timestamp) for s, _, _, _ in p1["executed_items"]}

    def _comp_row(v: VariantReport, badge: str) -> str:
        st = v.stats
        h1 = st["h1"]
        h2 = st["h2"]
        return (
            f"<tr>"
            f"<td><b>{html.escape(v.title)}</b> {badge}</td>"
            f"<td>{st['n']}</td>"
            f"<td>{st['wr']:.1f}%</td>"
            f"<td>{st['avg_pnl']:+.2f}%</td>"
            f"<td>{st['avg_r']:+.2f}R</td>"
            f"<td class='pos'><b>{st['sum_r']:+.1f}R</b></td>"
            f"<td>{st['pf']:.2f}</td>"
            f"<td>{st['max_streak']} 筆</td>"
            f"<td>{h1['n']} 筆 ({h1['sum_r']:+.1f}R)</td>"
            f"<td>{h2['n']} 筆 ({h2['sum_r']:+.1f}R)</td>"
            f"<td class='pos'><b>${v.port_2slot['eq_risk2']:,.0f} ({v.port_2slot['mdd_risk2']:.1f}%)</b></td>"
            f"<td>${v.port_2slot['eq_comp']:,.0f} ({v.port_2slot['mdd_comp']:.1f}%)</td>"
            f"</tr>"
        )

    trade_rows: List[str] = []
    cards_html: List[str] = []
    for idx, sig in enumerate(sniper.signals, start=1):
        key = (sig.symbol, sig.timestamp)
        ex = sniper.exit_map[key]
        pnl = float(ex["pnl_pct"])
        r_mult = float(ex["r_mult"])
        cls = "pos" if pnl > 0 else "neg"
        hold_h = (ex["exit_idx"] - sig.bar_idx) * 0.25
        slot_tag = []
        if key in exec2_keys:
            slot_tag.append("2槽✓")
        if key in exec1_keys:
            slot_tag.append("單倉✓")
        slot_str = " / ".join(slot_tag) if slot_tag else "重疊略過"
        card_rel = card_paths.get(key, "")

        trade_rows.append(
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td>{sig.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
            f"<td><b>{html.escape(sig.symbol)}</b></td>"
            f"<td>{html.escape(slot_str)}</td>"
            f"<td>{sig.entry_price:.6g}</td>"
            f"<td>{sig.risk_pct:.2f}%</td>"
            f"<td>{sig.ext200_pct:+.2f}%</td>"
            f"<td>{sig.qvm:.1f}x ({sig.vol_acc:.2f}x)</td>"
            f"<td>{sig.rsi6:.1f}</td>"
            f"<td>{html.escape(ex['reason'])} ({hold_h:.1f}h)</td>"
            f"<td class='{cls}'><b>{pnl:+.2f}%</b></td>"
            f"<td class='{cls}'><b>{r_mult:+.2f}R</b></td>"
            f"<td><a href='#card-{idx}'>復盤圖</a></td>"
            f"</tr>"
        )
        cards_html.append(
            f"<div class='card' id='card-{idx}'>"
            f"<div class='card-hd'>#{idx} {html.escape(sig.symbol)} | {sig.timestamp.strftime('%Y-%m-%d %H:%M')} TPE | "
            f"<span class='{cls}'>{pnl:+.2f}% ({r_mult:+.2f}R)</span> | {html.escape(ex['reason'])} ({hold_h:.1f}h) | {html.escape(slot_str)}</div>"
            f"<div class='card-sub'>進場 {sig.entry_price:.6g} | 風險 {sig.risk_pct:.2f}% | 距MA200 {sig.ext200_pct:+.2f}% | "
            f"8線糾結 {sig.spread8_pct:.2f}% | 量比 {sig.qvm:.1f}x (加速 {sig.vol_acc:.2f}x) | 主買 {sig.taker_buy_pct:.1f}% | RSI6 {sig.rsi6:.1f}</div>"
            f"<img src='{html.escape(card_rel)}' loading='lazy' alt='{html.escape(sig.symbol)}'/>"
            f"</div>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>MA200 均線粘合起漲策略 — 365 天回測與優化報表</title>
<style>
  body {{ margin:0; padding:24px; background:#0b1120; color:#e2e8f0; font-family:"Microsoft JhengHei",system-ui,sans-serif; }}
  .wrap {{ max-width:1420px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:24px; color:#f8fafc; }}
  .sub {{ color:#94a3b8; font-size:13.5px; margin-bottom:20px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:#111c33; border:1px solid #1e293b; border-radius:10px; padding:14px; }}
  .kpi .k {{ font-size:12px; color:#94a3b8; }}
  .kpi .v {{ font-size:21px; font-weight:700; margin-top:4px; color:#f8fafc; }}
  .panel {{ background:#111c33; border:1px solid #1e293b; border-radius:12px; padding:18px; margin-bottom:20px; }}
  .panel h2 {{ margin:0 0 12px; font-size:17px; color:#f8fafc; }}
  .insights {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:12px; }}
  .ins {{ background:#0f172a; border:1px solid #263449; border-radius:8px; padding:12px; font-size:13px; line-height:1.55; color:#cbd5e1; }}
  .ins b {{ color:#38bdf8; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:8px 9px; border-bottom:1px solid #1e293b; text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align:left; }}
  th {{ color:#94a3b8; font-weight:600; background:#0f172a; position:sticky; top:0; }}
  .pos {{ color:#4ade80; }}
  .neg {{ color:#f87171; }}
  .badge {{ display:inline-block; padding:2px 7px; border-radius:999px; font-size:11px; margin-left:6px; }}
  .b-best {{ background:#065f46; color:#a7f3d0; }}
  .b-base {{ background:#7f1d1d; color:#fecaca; }}
  .b-bal {{ background:#312e81; color:#c7d2fe; }}
  .hero-img {{ width:100%; border-radius:8px; border:1px solid #1e293b; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(560px,1fr)); gap:16px; }}
  .card {{ background:#111c33; border:1px solid #1e293b; border-radius:10px; padding:12px; }}
  .card-hd {{ font-size:14px; font-weight:700; margin-bottom:4px; }}
  .card-sub {{ font-size:12px; color:#94a3b8; margin-bottom:8px; }}
  .card img {{ width:100%; border-radius:6px; background:#fff; }}
  a {{ color:#38bdf8; text-decoration:none; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>MA200 均線粘合起漲策略 — 365 天全市場誠實回測與優化對比</h1>
  <div class="sub">資料範圍：{start_str} ~ {end_str} (TPE) | 時點滾動成交額標的池：{n_syms} 檔 Binance 永續合約 (15m K線) | 出場：1.2R 保本(同根保守檢查低點) + MA99 跟隨止盈 (最長 72h) | 已扣 {fee_pct:.2f}% 來回費滑 + 歷史/保守資金費率</div>

  <div class="kpis">
    <div class="kpi"><div class="k">★ 主指標：雙槽 {p2['risk_pct_per_trade']:g}% 以損定倉</div><div class="v pos">${p2['eq_risk2']:,.0f} (MDD {p2['mdd_risk2']:.1f}%)</div></div>
    <div class="kpi"><div class="k">雙槽 1x 真實無槓桿複利</div><div class="v pos">${p2['eq_spot_1x']:,.0f} (MDD {p2['mdd_spot_1x']:.1f}%)</div></div>
    <div class="kpi"><div class="k">Sniper 總訊號 / 勝率</div><div class="v pos">{st_s['n']} 筆 / {st_s['wr']:.1f}%</div></div>
    <div class="kpi"><div class="k">平均淨利 / 平均 R</div><div class="v pos">{st_s['avg_pnl']:+.2f}% / {st_s['avg_r']:+.2f}R</div></div>
    <div class="kpi"><div class="k">一年累計總 R / 盈虧比</div><div class="v pos">{st_s['sum_r']:+.1f}R / {st_s['pf']:.2f}</div></div>
    <div class="kpi"><div class="k">雙槽 3x [含強平/流動性上限·理論]</div><div class="v">${p2['eq_comp']:,.0f} (MDD {p2['mdd_comp']:.1f}%)</div></div>
  </div>

  <div class="panel">
    {bt_utils.render_oos_walkforward_html(sniper.oos_report, risk_pct=p2['risk_pct_per_trade']) if sniper.oos_report else ""}
  </div>

  <div class="panel">
    <h2>1. 優化前後與樣本內 (IS) → 樣本外盲測 (OOS) 對比</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>版本</th><th>訊號</th><th>勝率</th><th>平均1x淨利</th><th>平均R</th><th>總R</th><th>PF</th><th>最大連虧</th><th>IS 樣本內</th><th>OOS 樣本外</th><th>★雙槽{p2['risk_pct_per_trade']:g}%定倉(MDD)</th><th>雙槽3x理論對照(MDD)</th></tr>
        </thead>
        <tbody>
          {_comp_row(base, "<span class='badge b-base'>優化前</span>")}
          {_comp_row(sniper, "<span class='badge b-best'>穩健主推</span>")}
          {_comp_row(bal, "<span class='badge b-bal'>高頻平衡</span>")}
        </tbody>
      </table>
    </div>
    <div class="insights" style="margin-top:14px">
      <div class="ins"><b>去過擬合原則 ① 改採自然整數門檻</b><br/>移除貼著歷史單筆邊界的奇數小數（如 <code>88.5</code>、<code>-0.25%</code>、<code>1.15</code>），統一改用自然整數門檻：<code>vol_acc ≥ 1.0</code>（當根不縮量）、<code>RSI6 ≤ 90.0</code>、<code>qvm ≥ 3.5x</code>、<code>slp99 ≥ -0.50%</code>、<code>slp200 ≥ -0.30%</code>。</div>
      <div class="ins"><b>去過擬合原則 ② 移除 18h 微利釋放規則</b><br/>敏感度測試顯示 <code>scratch_bars=72</code> 僅因特定日期（01-01）釋放槽位才美化資金曲線，對總 R 無實質增益；預設改回 <code>scratch_bars=0</code>（純粹 1.2R 保本 + MA99 移動停利）。</div>
      <div class="ins"><b>實盤防護 ③ 僅用已收盤 K 線 + 雙槽分倉</b><br/>修正即時掃描預設 <code>drop_forming=True</code>（剔除未收盤當根 K 線以消除盤中訊號重繪），搭配雙槽並發將 3x 複利最大回撤壓低至 <b>{p2['mdd_comp']:.1f}%</b>。</div>
    </div>
  </div>

  <div class="panel">
    <h2>2. 365 天優化前後資金曲線與 Sniper 單筆績效分佈</h2>
    <img class="hero-img" src="equity_curve_365d.png" alt="365d Equity Curve"/>
  </div>

  <div class="panel">
    <h2>3. Sniper 狙擊模式全年交易明細 ({len(sniper.signals)} 筆)</h2>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr><th>#</th><th>訊號時間 (TPE)</th><th>幣種</th><th>實戰槽位</th><th>進場價</th><th>止損%</th><th>距MA200</th><th>爆量(加速)</th><th>RSI6</th><th>出場原因 (持倉)</th><th>1x 淨利</th><th>R 倍數</th><th>復盤圖</th></tr>
        </thead>
        <tbody>
          {"".join(trade_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>4. 全體 Sniper 交易 K 線復盤圖卡</h2>
    <div class="grid">
      {"".join(cards_html)}
    </div>
  </div>
</div>
</body>
</html>
"""
    html_path = out_dir / "index.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MA200 均線粘合起漲策略 365 天誠實回測與視覺化對比工具")
    p.add_argument("--cache-dir", type=Path, default=Path("breakout_output/cache_365d"))
    p.add_argument("--out-dir", type=Path, default=Path("breakout_output/backtest_365d_ma200"))
    p.add_argument("--max-hold", type=int, default=288)
    p.add_argument("--arm-r", type=float, default=1.2)
    p.add_argument("--fee-pct", type=float, default=0.12)
    p.add_argument("--min-vol", type=float, default=1_000_000.0, help="訊號當下前24h最低滾動成交額 (USDT)")
    p.add_argument("--top", type=int, default=0, help="時點滾動24h成交額前N名過濾 (0=全部符合min-vol標的)")
    p.add_argument("--risk-pct", type=float, default=2.0, help="主指標單筆風險百分比 (預設 2.0%%)")
    p.add_argument("--funding-rate-8h", type=float, default=0.03, help="無法抓取歷史費率時的保守估計資金費率 (%%/8h)")
    p.add_argument("--no-fetch-funding", action="store_true", help="不連網抓取歷史資金費率，直接使用保守估計值")
    p.add_argument("--max-bar-vol-pct", type=float, default=0.02, help="單筆倉位佔突破K線成交額上限比例 (預設 0.02)")
    p.add_argument("--no-liquidity-cap", action="store_true", help="停用倉位流動性上限")
    p.add_argument("--oos-date", type=str, default="2026-05-15", help="樣本外盲測切分日期 YYYY-MM-DD (空字串則依 --oos-ratio)")
    p.add_argument("--oos-ratio", type=float, default=0.50, help="樣本外盲測時間佔比")
    p.add_argument("--wf-folds", type=int, default=4, help="Walk-Forward 時間切片期數 (預設 4)")
    p.add_argument("--skip-cards", action="store_true", help="只輸出統計與資金曲線，不重繪單筆 K 線圖卡")
    return p.parse_args()


def main() -> int:
    t0 = time.time()
    args = parse_args()
    dfs, start_str, end_str = _load_cached_market(args.cache_dir)
    print(f"已載入 {len(dfs)} 檔 15m 快取資料 ({start_str} ~ {end_str})...")

    pit_index = bt_utils.PointInTimeVolumeIndex(dfs)
    funding_store = bt_utils.FundingRateStore(
        cache_dir=args.cache_dir,
        fallback_rate_8h_pct=args.funding_rate_8h,
        fetch_online=not args.no_fetch_funding,
        session=scanner.HTTP,
        base_url=scanner.BINANCE_FAPI,
    )
    window_start = min(df.index[0] for df in dfs.values())
    window_end = max(df.index[-1] for df in dfs.values())
    common_kw = dict(
        pit_index=pit_index,
        funding_store=funding_store,
        window_start=window_start,
        window_end=window_end,
        max_hold_bars=args.max_hold,
        arm_r=args.arm_r,
        fee_pct=args.fee_pct,
        min_vol_usd=args.min_vol,
        top_n=args.top,
        risk_pct=args.risk_pct,
        oos_date=args.oos_date,
        oos_ratio=args.oos_ratio,
        wf_folds=args.wf_folds,
        apply_liquidity_cap=not args.no_liquidity_cap,
        max_bar_vol_share=args.max_bar_vol_pct,
    )

    variants = {
        "baseline": _run_variant(
            dfs,
            "baseline",
            "Baseline v1 (優化前)",
            preset_override=BASELINE_V1_PARAMS,
            preset_name="sniper",
            scratch_bars=0,
            **common_kw,
        ),
        "sniper": _run_variant(
            dfs,
            "sniper",
            "Sniper 狙擊模式 (優化主推)",
            preset_override=None,
            preset_name="sniper",
            **common_kw,
        ),
        "balanced": _run_variant(
            dfs,
            "balanced",
            "Balanced 平衡模式",
            preset_override=None,
            preset_name="balanced",
            **common_kw,
        ),
    }

    print(f"\n=== 365 天優化前後誠實回測總覽 (已扣 {args.fee_pct:.2f}% 費滑 + 資金費率) ===")
    for key in ("baseline", "sniper", "balanced"):
        v = variants[key]
        st = v.stats
        p2 = v.port_2slot
        print(
            f"[{v.title:<24}] n={st['n']:2d} | WR={st['wr']:5.1f}% | AvgP={st['avg_pnl']:+6.2f}% | "
            f"AvgR={st['avg_r']:+5.2f}R | SumR={st['sum_r']:+6.1f}R | PF={st['pf']:5.2f} | "
            f"IS={st['h1']['n']}({st['h1']['sum_r']:+.1f}R) OOS={st['h2']['n']}({st['h2']['sum_r']:+.1f}R) | "
            f"★2槽{args.risk_pct:g}%定倉=${p2['eq_risk2']:,.0f}(MDD {p2['mdd_risk2']:.1f}%) | "
            f"1x複利=${p2['eq_spot_1x']:,.0f} | 3x理論=${p2['eq_comp']:,.0f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eq_path = args.out_dir / "equity_curve_365d.png"
    plot_equity_comparison(variants, len(dfs), start_str, end_str, eq_path)

    card_paths: Dict[Tuple[str, datetime], str] = {}
    if not args.skip_cards:
        card_paths = render_trade_cards(dfs, variants["sniper"], args.out_dir / "cards")
    html_path = write_html_report(args.out_dir, variants, card_paths, len(dfs), start_str, end_str, args.fee_pct)

    print(f"\n[Sniper 全部 {len(variants['sniper'].signals)} 筆交易清單]")
    scanner.print_signals(variants["sniper"].signals, exit_map=variants["sniper"].exit_map)
    print(f"\n完成! 耗時 {time.time() - t0:.1f}s | 資金曲線: {eq_path} | HTML 報表: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
