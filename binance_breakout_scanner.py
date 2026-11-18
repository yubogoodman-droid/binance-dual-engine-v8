#!/usr/bin/env python3
"""幣安永續合約「起漲點」偵測與即時掃描器 v2 (優化版)

對齊手機幣安看盤設定：
- 均線系統：MA7, MA14, MA25, MA30, MA60, MA99, MA120, MA200
- 成交量均線：VOL MA5, MA10, MA20

v2 實盤回測優化重點：
1. 【排除反向/波動率衰減標的】：自動排除 SOXS / SQQQ / TBT / TZA / UVXY 等反向損耗商品。
2. 【K線收盤強度濾網 (CLV >= 0.68)】：起漲點火 K 線收盤價必須收在當根高低點 68% 以上位置，過濾衝高回落假突破。
3. 【長期均線築底確認 (MA99 斜率保護)】：過濾 MA99 仍在劇烈下殺 (> -2%) 的空頭反抽，確保底部已走平築底。
4. 【量價壓縮提純】：箱體壓縮 <= 9.5%、均線糾結 <= 4.5%、底部點火爆量 >= 4.5x（或透過 CLI 自訂）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("binance_breakout_scanner")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

TPE = ZoneInfo("Asia/Taipei")
BASE = "https://fapi.binance.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA, "Accept": "application/json"})
_RATE_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0

EXCLUDED_SYMBOLS = frozenset({
    "SQQQUSDT",
    "SOXSUSDT",
    "SKDDUSDT",
    "TBTUSDT",
    "TZAUSDT",
    "UVXYUSDT",
    "USDCUSDT",
    "FDUSDUSDT",
    "AMDUSDT",
    "NVDAUSDT",
    "TSLAUSDT",
    "AAPLUSDT",
    "MSTRUSDT",
    "COINUSDT",
    "TQQQUSDT",
    "SPCXUSDT",
    "SNDKUSDT",
    "SNXXUSDT",
    "SOXLUSDT",
    "AVGOUSDT",
    "ONDSUSDT",
    "CRCLUSDT",
    "GOOGLUSDT",
    "METAUSDT",
    "AMZNUSDT",
    "MSFTUSDT",
    "PLTRUSDT",
    "HOODUSDT",
    "SMCIUSDT",
    "BABAUSDT",
    "SKHYNIXUSDT",
    "SKHYUSDT",
    "NBISUSDT",
    "KORUUSDT",
    "MUUSDT",
    "MUUUSDT",
    "QQQUSDT",
    "SPYUSDT",
    "CLUSDT",
    "BZUSDT",
    "XAUUSDT",
    "XAGUSDT",
})

# 對齊手機幣安 App 均線顏色
MA_SPECS = [
    (7, "#f0b90b", 1.2, "MA(7)"),     # 黃
    (14, "#26c6da", 1.1, "MA(14)"),   # 淺藍
    (25, "#ec407a", 1.7, "MA(25)"),   # 粉紅
    (30, "#2962ff", 1.1, "MA(30)"),   # 寶藍
    (60, "#2e7d32", 1.2, "MA(60)"),   # 深綠
    (99, "#7e57c2", 1.3, "MA(99)"),   # 紫
    (120, "#66bb6a", 1.2, "MA(120)"), # 淺綠
    (200, "#8e24aa", 1.4, "MA(200)"), # 酒紅
]

INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}


@dataclass
class BreakoutSignal:
    symbol: str
    interval: str
    signal_type: str          # "BOTTOM_BREAKOUT" | "MA25_CONTINUATION"
    bar_idx: int
    timestamp: datetime
    entry_price: float
    stop_price: float
    target_2r: float
    target_4r: float
    vol_ratio: float          # 當根成交量 / 前20根均量
    bar_gain_pct: float       # 當根K線漲幅 (%)
    ma_spread_pct: float      # 起漲前均線糾結度 (%)
    base_range_pct: float     # 起漲前橫盤振幅 (%)
    clv: float                # 收盤強度 (Close - Low) / (High - Low)
    max_gain_after_pct: float # 訊號後最高漲幅 (%)
    max_r_after: float        # 訊號後最高盈虧比 (R)
    note: str


def get_json(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Any:
    global _COOLDOWN_UNTIL
    last_err = None
    for i in range(retries):
        wait_cd = 0.0
        with _RATE_LOCK:
            now_m = time.monotonic()
            if _COOLDOWN_UNTIL > now_m:
                wait_cd = _COOLDOWN_UNTIL - now_m
        if wait_cd > 0:
            time.sleep(wait_cd)
        try:
            r = SESSION.get(BASE + path, params=params, timeout=20)
            used_w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if used_w and used_w.isdigit() and int(used_w) >= 1950:
                with _RATE_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + 10.0)
            if r.status_code in (418, 429):
                retry_hdr = r.headers.get("Retry-After")
                wait_s = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else 2.0 * (2 ** i)
                with _RATE_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + wait_s)
                logger.warning(
                    "Binance API 觸發限流 (HTTP %s, %s)，等待 %.1fs 後重試 (%d/%d)",
                    r.status_code,
                    path,
                    wait_s,
                    i + 1,
                    retries,
                )
                time.sleep(wait_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            wait_s = 0.5 * (2 ** i)
            if i + 1 < retries:
                logger.warning("Binance API 請求失敗 (%s): %s，%.1fs 後重試 (%d/%d)", path, exc, wait_s, i + 1, retries)
                time.sleep(wait_s)
    raise RuntimeError(f"API request failed ({path}): {last_err}")


def fetch_top_usdt_perps(top_n: int = 80, min_quote_vol: float = 10_000_000, include_tradfi: bool = False) -> List[str]:
    """取得幣安 24h 成交額前 N 名的純加密貨幣 USDT 永續合約清單（預設排除美股休市假突破與反向合約）。"""
    info = get_json("/fapi/v1/exchangeInfo")
    valid_syms = set()
    allowed_types = ("PERPETUAL", "TRADIFI_PERPETUAL") if include_tradfi else ("PERPETUAL",)
    for s in info.get("symbols", []):
        sym = s.get("symbol", "")
        if sym in EXCLUDED_SYMBOLS:
            continue
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("contractType") in allowed_types
        ):
            valid_syms.add(sym)

    tickers = get_json("/fapi/v1/ticker/24hr")
    ranked = []
    for t in tickers:
        sym = t.get("symbol", "")
        if sym not in valid_syms:
            continue
        qv = float(t.get("quoteVolume") or 0.0)
        if qv >= min_quote_vol:
            ranked.append((qv, sym))
    ranked.sort(reverse=True)
    return [sym for _, sym in ranked[:top_n]]


def fetch_klines(symbol: str, interval: str = "15m", days: int = 5, drop_forming: bool = True) -> pd.DataFrame:
    """拉取幣安永續合約 K 線並計算 MA7~MA200 與量能指標（預設剔除未收盤當根 K 線避免重繪）。"""
    bar_ms = INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = min(1500, max(250, days * bars_per_day + 200 + (1 if drop_forming else 0)))

    raw = get_json(
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": need},
    )
    if not raw:
        return pd.DataFrame()

    now_ms = int(time.time() * 1000)
    while drop_forming and raw and int(raw[-1][0]) + bar_ms > now_ms:
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

    for period, _, _, _ in MA_SPECS:
        df[f"MA{period}"] = df["Close"].rolling(period, min_periods=period).mean()
    df["MA240"] = df["Close"].rolling(240, min_periods=240).mean()

    df["VOL_MA5"] = df["Volume"].rolling(5, min_periods=3).mean()
    df["VOL_MA10"] = df["Volume"].rolling(10, min_periods=5).mean()
    df["VOL_MA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["VOL_MA20_PREV"] = df["Volume"].shift(1).rolling(20, min_periods=10).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20_PREV"].replace(0, np.nan)
    return df


def detect_coiled_ma200_signals(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
    *,
    min_bars_below_ma200: int = 80,
    coil_lookback: int = 8,
    max_coil_7_120_pct: float = 1.80,
    max_coil_long_pct: float = 1.20,
    slope_lookback: int = 16,
    min_ma60_slope_pct: float = 0.00,
    min_ma200_slope_pct: float = -0.60,
    min_ext_ma200_pct: float = 0.10,
    max_ext_ma200_pct: float = 3.50,
    min_bar_chg_pct: float = 0.80,
    min_body_ratio: float = 0.55,
    min_vol_ratio: float = 2.00,
    max_vol_ratio: float = 8.00,
    min_green_streak: int = 2,
    min_above_ma60_bars: int = 6,
    min_mom4_pct: float = 1.60,
    max_upper_wick_ratio: float = 0.24,
    min_vol_vs_12bar_max: float = 1.10,
    min_gap_ma200_long_pct: float = 0.40,
    max_gap_ma200_long_pct: float = 3.20,
    follow_min_bars: int = 1,
    follow_max_bars: int = 4,
    follow_min_ext_ma200_pct: float = 2.20,
    follow_min_vol_ratio: float = 2.00,
    follow_min_bar_chg_pct: float = 1.00,
    follow_max_upper_wick_ratio: float = 0.30,
) -> List[BreakoutSignal]:
    """引擎 B（牛來專案）：底部七線黏帶 · 倍量穿 MA200 · 第二拍扇形展開強確認。"""
    if df.empty or len(df) <= 245:
        return []

    c = df["Close"].to_numpy(float)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    v = df["Volume"].to_numpy(float)
    timestamps = df.index

    def _sma(a: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(a), np.nan)
        if len(a) >= n:
            out[n - 1 :] = np.convolve(a, np.ones(n) / n, mode="valid")
        return out

    m = {n: df[f"MA{n}"].to_numpy(float) if f"MA{n}" in df.columns else _sma(c, n) for n in (7, 14, 25, 30, 60, 99, 120, 200)}
    m240 = df["MA240"].to_numpy(float) if "MA240" in df.columns else _sma(c, 240)
    v20 = df["VOL_MA20"].to_numpy(float) if "VOL_MA20" in df.columns else _sma(v, 20)

    n_bars = len(c)
    m200 = m[200]
    valid = ~np.isnan(m200[210:])
    cross_mask = valid & (c[210:] > m200[210:]) & ((c[209:-1] <= m200[209:-1]) | (c[208:-2] <= m200[208:-2]))
    cand_indices = np.flatnonzero(cross_mask) + 210

    signals: List[BreakoutSignal] = []
    for idx in cand_indices:
        i = int(idx)
        if i < max(210, coil_lookback + 200, slope_lookback + 200) or i >= n_bars:
            continue
        needed = [m[p][i] for p in (7, 14, 25, 30, 60, 99, 120, 200)] + [
            m[60][i - slope_lookback],
            m[200][i - slope_lookback],
            v20[i],
        ]
        if np.isnan(needed).any() or v20[i] <= 0:
            continue

        # 1. 收盤正式從下方突破站上 MA200 (clean_first_cross)
        if not (c[i] > m200[i] and c[i - 1] <= m200[i - 1]):
            continue

        ref_idx = i - 1
        bars_below = 0
        for k in range(ref_idx, max(0, ref_idx - 200), -1):
            if np.isnan(m200[k]) or c[k] > m200[k]:
                break
            bars_below += 1
        if bars_below < min_bars_below_ma200:
            continue

        # 2. 短中期均線多頭排列，允許突破當根 MA25 在長均簇內
        m7_i, m14_i, m25_i, m30_i = m[7][i], m[14][i], m[25][i], m[30][i]
        m60_i, m99_i, m120_i, m200_i = m[60][i], m[99][i], m[120][i], m200[i]
        long_top = max(m60_i, m99_i, m120_i)
        long_bot = min(m60_i, m99_i, m120_i)
        if not (m7_i > m14_i > m25_i and m7_i > long_top and m25_i >= long_bot):
            continue

        # 3. 突破前 8 根內 MA7..MA120 七線黏帶，且與 MA200 呈碗底蓄力空間
        coil_candidates = []
        for k in range(i - coil_lookback, i):
            vals7 = [m[p][k] for p in (7, 14, 25, 30, 60, 99, 120)]
            if np.isnan(vals7).any():
                continue
            coil_candidates.append((max(vals7) / min(vals7) - 1.0) * 100.0)
        if not coil_candidates:
            continue
        coil_7_120_pct = float(min(coil_candidates))
        coil_long_pct = float((long_top / long_bot - 1.0) * 100.0)
        gap_ma200_long_pct = float((m200_i / long_top - 1.0) * 100.0)
        if coil_7_120_pct > max_coil_7_120_pct or coil_long_pct > max_coil_long_pct:
            continue
        if not (min_gap_ma200_long_pct <= gap_ma200_long_pct <= max_gap_ma200_long_pct):
            continue

        # 4. 長均線走平翻揚
        ma60_slope_pct = float((m60_i / m[60][i - slope_lookback] - 1.0) * 100.0)
        ma200_slope_pct = float((m200_i / m[200][i - slope_lookback] - 1.0) * 100.0)
        if ma60_slope_pct < min_ma60_slope_pct or ma200_slope_pct < min_ma200_slope_pct:
            continue

        # 5. 突破前已站穩 MA60、連紅階梯推升、4根動能
        above60_cnt = sum(1 for k in range(i - 12, i) if not np.isnan(m[60][k]) and c[k] >= m[60][k])
        if above60_cnt < min_above_ma60_bars:
            continue
        green_streak = 0
        for k in range(i, max(0, i - 10), -1):
            if c[k] >= o[k]:
                green_streak += 1
            else:
                break
        if green_streak < min_green_streak:
            continue
        mom4_pct = float((c[i] / c[i - 4] - 1.0) * 100.0)
        if mom4_pct < min_mom4_pct:
            continue

        # 6. 第一拍紅 K 實體、無長上影線、乖離與倍量過濾 (2.0x ~ 8.0x)
        bar_chg_pct = float((c[i] / o[i] - 1.0) * 100.0)
        rng = max(h[i] - l[i], 1e-12)
        body_ratio = float((c[i] - o[i]) / rng)
        upper_wick_ratio = float((h[i] - c[i]) / rng)
        ext_ma200_pct = float((c[i] / m200_i - 1.0) * 100.0)
        vol_ratio = float(v[i] / v20[i])
        vol_vs_12bar_max = float(v[i] / max(float(np.max(v[i - 12 : i])), 1e-12))

        if bar_chg_pct < min_bar_chg_pct or body_ratio < min_body_ratio:
            continue
        if upper_wick_ratio > max_upper_wick_ratio:
            continue
        if not (min_ext_ma200_pct <= ext_ma200_pct <= max_ext_ma200_pct):
            continue
        if vol_ratio < min_vol_ratio or vol_ratio > max_vol_ratio or vol_vs_12bar_max < min_vol_vs_12bar_max:
            continue

        stop_price = float(min(long_bot, l[max(0, i - 3) : i + 1].min()))

        # 7. 尋找第二拍扇形展開強確認 (j in i+1 .. i+4)
        for j in range(i + follow_min_bars, min(n_bars, i + follow_max_bars + 1)):
            if np.isnan([m[7][j], m[14][j], m[25][j], m[30][j], m200[j], v20[j]]).any() or v20[j] <= 0:
                continue
            if any(c[k] <= m200[k] for k in range(i, j + 1)):
                continue
            if not (c[j] > o[j] and c[j] > c[i]):
                continue
            rng_j = max(h[j] - l[j], 1e-12)
            if (h[j] - c[j]) / rng_j > follow_max_upper_wick_ratio:
                continue
            if not (m[7][j] > m[14][j] > m[25][j] > m[30][j] > m200[j] or (m[7][j] > m[14][j] > m[25][j] > m[30][j] and c[j] > c[i])):
                continue
            ext200_j = float((c[j] / m200[j] - 1.0) * 100.0)
            vr_j = float(v[j] / v20[j])
            if ext200_j < follow_min_ext_ma200_pct or vr_j < follow_min_vol_ratio:
                continue
            if not np.isnan(m240[j]) and c[j] <= m240[j]:
                continue
            f_chg_pct = float((c[j] - o[j]) / max(o[j], 1e-12) * 100.0)
            if f_chg_pct < follow_min_bar_chg_pct:
                break

            entry_px = float(c[j])
            risk = entry_px - stop_price
            if risk <= 0:
                break
            future_high = float(np.max(h[j:])) if j < n_bars else entry_px
            max_gain_after = (future_high - entry_px) / entry_px * 100.0
            max_r = (future_high - entry_px) / risk
            clv_j = float((c[j] - l[j]) / rng_j)

            signals.append(
                BreakoutSignal(
                    symbol=symbol,
                    interval=interval,
                    signal_type="COILED_MA200_STAGE2",
                    bar_idx=j,
                    timestamp=timestamps[j],
                    entry_price=entry_px,
                    stop_price=stop_price,
                    target_2r=entry_px + 2.0 * risk,
                    target_4r=entry_px + 4.0 * risk,
                    vol_ratio=vol_ratio,
                    bar_gain_pct=f_chg_pct,
                    ma_spread_pct=coil_7_120_pct,
                    base_range_pct=gap_ma200_long_pct,
                    clv=clv_j,
                    max_gain_after_pct=max_gain_after,
                    max_r_after=max_r,
                    note=(
                        f"牛來水下築底({bars_below}根) + 7線黏帶({coil_7_120_pct:.2f}%) + 碗口({gap_ma200_long_pct:.2f}%) "
                        f"→ 第一拍 {vol_ratio:.1f}x 穿MA200 + 第二拍(+{f_chg_pct:.1f}%, 量{vr_j:.1f}x)扇形展開強確認"
                    ),
                )
            )
            break

    return signals


def detect_signals(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    *,
    consol_bars: int = 20,
    max_ma_spread_pct: float = 4.5,
    max_base_range_pct: float = 9.5,
    min_vol_ratio: float = 5.5,
    min_bar_gain_pct: float = 3.5,
    max_bar_gain_pct: float = 22.0,
    min_clv: float = 0.68,
    max_ma99_drop_pct: float = 2.0,
    max_pre_med_vol_ratio: float = 1.40,
    max_ext_ma25_pct: float = 9.25,
    max_ma25_vs_ma200_pct: float = 4.5,
    min_breakout_margin_pct: float = 2.65,
    cooldown_bars: int = 12,
    include_continuation: bool = False,
) -> List[BreakoutSignal]:
    """在 K 線序列中偵測優化後的「第一買點：底部起漲突破」與「第二買點：回踩 MA25 二次起漲」。"""
    if df.empty or len(df) < consol_bars + 25:
        return []

    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    vol_ratios = df["VOL_RATIO"].to_numpy(float)
    ma7 = df["MA7"].to_numpy(float)
    ma14 = df["MA14"].to_numpy(float)
    ma25 = df["MA25"].to_numpy(float)
    ma30 = df["MA30"].to_numpy(float)
    ma60 = df["MA60"].to_numpy(float)
    ma99 = df["MA99"].to_numpy(float)
    ma120 = df["MA120"].to_numpy(float) if "MA120" in df.columns else ma99
    ma200 = df["MA200"].to_numpy(float) if "MA200" in df.columns else ma99
    timestamps = df.index

    signals: List[BreakoutSignal] = []
    last_bottom_idx = -999
    last_cont_idx = -999

    for i in range(consol_bars + 10, len(df)):
        c = closes[i]
        o = opens[i]
        h = highs[i]
        l = lows[i]
        vr = vol_ratios[i] if np.isfinite(vol_ratios[i]) else 0.0

        if c <= o or o <= 0 or h <= l:
            continue

        bar_gain_pct = (c - o) / o * 100.0
        clv = (c - l) / (h - l)

        # 濾網 1：K線收盤強度 (CLV >= min_clv)，避免衝高回落留長上影線的假突破
        if clv < min_clv:
            continue

        # 濾網 1b：不追高乖離 (當根收盤價偏離 MA25 不可超過 max_ext_ma25_pct，避免買在短線乖離極限)
        if np.isfinite(ma25[i]) and ma25[i] > 0:
            ext_ma25_pct = (c - ma25[i]) / ma25[i] * 100.0
            if ext_ma25_pct > max_ext_ma25_pct:
                continue

        # =====================================================================
        # 訊號 1：【底部均線糾結 + 縮量沉澱 + 爆量長紅起漲點 (BOTTOM_BREAKOUT)】
        # =====================================================================
        if i - last_bottom_idx >= cooldown_bars:
            # 濾網 2：確認長期均線 MA99 已走平築底，且 MA25 已貼近或站上 MA99 (非空頭崩跌中繼)
            ma99_ok = True
            if np.isfinite(ma99[i]) and np.isfinite(ma99[i - 10]) and ma99[i - 10] > 0:
                ma99_change_pct = (ma99[i] - ma99[i - 10]) / ma99[i - 10] * 100.0
                if ma99_change_pct < -max_ma99_drop_pct:
                    ma99_ok = False
            if np.isfinite(ma99[i]) and np.isfinite(ma25[i]) and ma25[i] < ma99[i] * 0.985:
                ma99_ok = False

            # 濾網 3：起漲前 10 根 K 線必須處於「縮量沉澱」狀態 (成交量中位數 <= max_pre_med_vol_ratio)
            prev10_vrs = vol_ratios[max(0, i - 10) : i]
            med_prev_vr = float(np.nanmedian(prev10_vrs)) if len(prev10_vrs) else 1.0
            if not np.isfinite(med_prev_vr) or med_prev_vr > max_pre_med_vol_ratio:
                ma99_ok = False

            # 濾網 4 (v6 真底部長期均線位置)：排除已經在 MA200 上方半空中 (> max_ma25_vs_ma200_pct) 的假底部回踩
            if np.isfinite(ma25[i - 1]) and np.isfinite(ma200[i - 1]) and ma200[i - 1] > 0:
                ma25_vs_ma200_pct = (ma25[i - 1] - ma200[i - 1]) / ma200[i - 1] * 100.0
                if ma25_vs_ma200_pct > max_ma25_vs_ma200_pct:
                    ma99_ok = False

            # 濾網 5 (v6 排除前 6 根內已有爆量長針洗盤)：避免消息面雙巴盤或主力剛插針出貨完的二次假拉抬
            for k in range(max(0, i - 6), i):
                rng_k = (highs[k] - lows[k]) / lows[k] * 100.0 if lows[k] > 0 else 0.0
                vr_k = vol_ratios[k] if np.isfinite(vol_ratios[k]) else 0.0
                if rng_k > 4.0 and vr_k > 6.0:
                    ma99_ok = False
                    break

            # 濾網 6 (v6 排除階梯緩漲末端趕頂)：若前 14 根 K 線 [i-16:i-2] 全數已站上 MA25 且 MA7>MA25，且 MA60 已明顯上揚 (>0.65%)，屬連漲末端而非底部起漲
            if i >= 25 and np.isfinite(ma60[i - 1]) and np.isfinite(ma60[i - 25]) and ma60[i - 25] > 0:
                sub_c = closes[i - 16 : i - 2]
                sub_ma7 = ma7[i - 16 : i - 2]
                sub_ma25 = ma25[i - 16 : i - 2]
                bull_bars = int(np.sum((sub_c > sub_ma25) & (sub_ma7 > sub_ma25)))
                ma60_slp25 = (ma60[i - 1] - ma60[i - 25]) / ma60[i - 25] * 100.0
                if bull_bars >= 14 and ma60_slp25 > 0.65:
                    ma99_ok = False

            # 濾網 7 (v7 長期均線雙雙下彎空頭壓制排除)：確認 MA200 已有效且 MA99 與 MA200 不可同時處於下彎空頭趨勢
            if not np.isfinite(ma200[i]) or ma200[i] <= 0:
                ma99_ok = False
            elif i >= 20 and np.isfinite(ma99[i - 20]) and np.isfinite(ma200[i - 20]) and ma99[i - 20] > 0 and ma200[i - 20] > 0:
                ma99_slp20 = (ma99[i] - ma99[i - 20]) / ma99[i - 20] * 100.0
                ma200_slp20 = (ma200[i] - ma200[i - 20]) / ma200[i - 20] * 100.0
                if ma99_slp20 < -0.10 and ma200_slp20 < -0.10:
                    ma99_ok = False

            prev_slice = slice(i - consol_bars, i)
            prev_high = float(np.max(highs[prev_slice]))
            prev_low = float(np.min(lows[prev_slice]))
            base_range_pct = (prev_high - prev_low) / prev_low * 100.0 if prev_low > 0 else 999.0
            brk_margin_pct = (c - prev_high) / prev_high * 100.0 if prev_high > 0 else 0.0

            ma_candidates = [ma7[i - 1], ma14[i - 1], ma25[i - 1]]
            if np.isfinite(ma60[i - 1]):
                ma_candidates.append(ma60[i - 1])
            if np.isfinite(ma99[i - 1]):
                ma_candidates.append(ma99[i - 1])
            finite_mas = [float(x) for x in ma_candidates if np.isfinite(x) and x > 0]

            if ma99_ok and len(finite_mas) >= 3:
                ma_max = max(finite_mas)
                ma_min = min(finite_mas)
                ma_spread_pct = (ma_max - ma_min) / ma_min * 100.0

                current_mas = [x for x in (ma7[i], ma14[i], ma25[i], ma60[i], ma99[i], ma120[i], ma200[i]) if np.isfinite(x)]
                above_all_mas = all(c > m for m in current_mas)

                if (
                    ma_spread_pct <= max_ma_spread_pct
                    and base_range_pct <= max_base_range_pct
                    and vr >= min_vol_ratio
                    and min_bar_gain_pct <= bar_gain_pct <= max_bar_gain_pct
                    and brk_margin_pct >= min_breakout_margin_pct
                    and above_all_mas
                ):
                    last_bottom_idx = i
                    # 濾網 8 (v8 174檔全市場實戰優化：8線糾結 + 排除插針出貨 + 底部黃金收斂/極窄箱體旱地拔蔥)
                    mas8_prev = [
                        ma7[i - 1],
                        ma14[i - 1],
                        ma25[i - 1],
                        ma30[i - 1],
                        ma60[i - 1],
                        ma99[i - 1],
                        ma120[i - 1],
                        ma200[i - 1],
                    ]
                    if any(not np.isfinite(x) or x <= 0 for x in mas8_prev):
                        continue
                    spread8 = (max(mas8_prev) - min(mas8_prev)) / min(mas8_prev) * 100.0
                    ma25_200 = abs(ma25[i - 1] - ma200[i - 1]) / ma200[i - 1] * 100.0
                    ma60_slp16 = (ma60[i] - ma60[i - 16]) / ma60[i - 16] * 100.0 if i >= 16 and ma60[i - 16] > 0 else 0.0
                    ma200_slp16 = (ma200[i] - ma200[i - 16]) / ma200[i - 16] * 100.0 if i >= 16 and ma200[i - 16] > 0 else 0.0

                    if not (
                        10.0 <= vr <= 34.0
                        and spread8 <= 3.40
                        and base_range_pct <= 6.00
                        and bar_gain_pct <= 9.00
                        and ma25_200 >= 0.80
                    ):
                        continue

                    cond_a1 = ma60_slp16 >= 0.12 and ma200_slp16 <= -0.02 and brk_margin_pct >= 2.70
                    cond_a2 = spread8 <= 2.50 and base_range_pct <= 2.70 and brk_margin_pct >= 4.60
                    if not (cond_a1 or cond_a2):
                        continue

                    stop_price = min(l, prev_low) * 0.995
                    risk = max(c - stop_price, c * 0.015)
                    future_high = float(np.max(highs[i:])) if i < len(df) else c
                    max_gain_after = (future_high - c) / c * 100.0
                    max_r = (future_high - c) / risk if risk > 0 else 0.0

                    signals.append(
                        BreakoutSignal(
                            symbol=symbol,
                            interval=interval,
                            signal_type="BOTTOM_BREAKOUT",
                            bar_idx=i,
                            timestamp=timestamps[i],
                            entry_price=c,
                            stop_price=stop_price,
                            target_2r=c + 2.0 * risk,
                            target_4r=c + 4.0 * risk,
                            vol_ratio=vr,
                            bar_gain_pct=bar_gain_pct,
                            ma_spread_pct=spread8,
                            base_range_pct=base_range_pct,
                            clv=clv,
                            max_gain_after_pct=max_gain_after,
                            max_r_after=max_r,
                            note=(
                                f"底部8線糾結({spread8:.1f}%) + 箱體壓縮({base_range_pct:.1f}%) "
                                f"+ 實體飽滿({clv*100:.0f}%) → 爆量 {vr:.1f}x 長紅(+{bar_gain_pct:.1f}%)突破起漲"
                            ),
                        )
                    )
                    continue

        # =====================================================================
        # 訊號 2：【主升段空中加油：回踩 MA25 不破二次起漲 (MA25_CONTINUATION)】
        # =====================================================================
        if (
            include_continuation
            and i - last_cont_idx >= cooldown_bars
            and i - last_bottom_idx >= 8
            and np.isfinite(ma25[i])
            and np.isfinite(ma25[i - 5])
            and np.isfinite(ma7[i])
        ):
            ma25_slope_pct = (ma25[i] - ma25[i - 5]) / ma25[i - 5] * 100.0
            trend_ok = ma25_slope_pct >= 2.2 and (not np.isfinite(ma99[i]) or ma25[i] > ma99[i] * 1.06)

            if trend_ok:
                recent_lows = lows[i - 3 : i + 1]
                recent_closes = closes[i - 3 : i + 1]
                recent_ma25 = ma25[i - 3 : i + 1]
                dist_to_ma25_pct = float(np.min((recent_lows - recent_ma25) / recent_ma25 * 100.0))
                closes_held_ma25 = bool(np.all(recent_closes >= recent_ma25 * 0.99))
                had_vol_dryup = bool(np.nanmin(vol_ratios[max(0, i - 6) : i]) < 0.80)

                prev6_close_high = float(np.max(closes[i - 6 : i]))
                reclaim_trigger = (
                    c > ma7[i]
                    and c > prev6_close_high
                    and vr >= 1.25
                    and bar_gain_pct >= 3.0
                )

                if -3.5 <= dist_to_ma25_pct <= 2.5 and closes_held_ma25 and had_vol_dryup and reclaim_trigger:
                    pullback_low = float(np.min(lows[i - 4 : i + 1]))
                    stop_price = min(pullback_low, ma30[i] if np.isfinite(ma30[i]) else pullback_low) * 0.99
                    risk = max(c - stop_price, c * 0.02)
                    future_high = float(np.max(highs[i:])) if i < len(df) else c
                    max_gain_after = (future_high - c) / c * 100.0
                    max_r = (future_high - c) / risk if risk > 0 else 0.0

                    signals.append(
                        BreakoutSignal(
                            symbol=symbol,
                            interval=interval,
                            signal_type="MA25_CONTINUATION",
                            bar_idx=i,
                            timestamp=timestamps[i],
                            entry_price=c,
                            stop_price=stop_price,
                            target_2r=c + 2.0 * risk,
                            target_4r=c + 4.0 * risk,
                            vol_ratio=vr,
                            bar_gain_pct=bar_gain_pct,
                            ma_spread_pct=abs(dist_to_ma25_pct),
                            base_range_pct=0.0,
                            clv=clv,
                            max_gain_after_pct=max_gain_after,
                            max_r_after=max_r,
                            note=(
                                f"多頭縮量回踩 MA25 不破(誤差 {dist_to_ma25_pct:+.1f}%) "
                                f"→ 放量 {vr:.1f}x 突破前高二次起漲"
                            ),
                        )
                    )
                    last_cont_idx = i

    return signals


def plot_symbol_signals(
    df: pd.DataFrame,
    signals: List[BreakoutSignal],
    symbol: str,
    interval: str,
    out_path: Path,
    exit_info: Optional[Dict[datetime, Dict[str, Any]]] = None,
) -> None:
    """繪製仿幣安風格的 K 線 + 8 條均線 + 成交量 + 起漲點(及出場點)標記圖。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plot_df = df.tail(180).copy()
    if plot_df.empty:
        return

    fig, (ax_price, ax_vol) = plt.subplots(
        2,
        1,
        figsize=(15, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.06},
        facecolor="#ffffff",
    )

    x = np.arange(len(plot_df))
    opens = plot_df["Open"].to_numpy(float)
    highs = plot_df["High"].to_numpy(float)
    lows = plot_df["Low"].to_numpy(float)
    closes = plot_df["Close"].to_numpy(float)
    vols = plot_df["Volume"].to_numpy(float)
    times = plot_df.index

    up_color = "#0ecb81"
    down_color = "#f6465d"

    for idx in range(len(plot_df)):
        is_up = closes[idx] >= opens[idx]
        color = up_color if is_up else down_color
        ax_price.plot([idx, idx], [lows[idx], highs[idx]], color=color, linewidth=1.0, zorder=2)
        body_low = min(opens[idx], closes[idx])
        body_h = max(abs(closes[idx] - opens[idx]), (highs[idx] - lows[idx]) * 0.02, 1e-8)
        ax_price.add_patch(
            Rectangle((idx - 0.32, body_low), 0.64, body_h, facecolor=color, edgecolor=color, zorder=3)
        )
        ax_vol.bar(idx, vols[idx], width=0.68, color=color, alpha=0.85, zorder=2)

    for period, color, width, label in MA_SPECS:
        col = f"MA{period}"
        if col in plot_df.columns and plot_df[col].notna().any():
            ax_price.plot(x, plot_df[col].to_numpy(float), color=color, linewidth=width, label=label, zorder=4)

    if "VOL_MA5" in plot_df.columns:
        ax_vol.plot(x, plot_df["VOL_MA5"].to_numpy(float), color="#f0b90b", linewidth=1.2, label="VOL MA(5)")
    if "VOL_MA10" in plot_df.columns:
        ax_vol.plot(x, plot_df["VOL_MA10"].to_numpy(float), color="#7e57c2", linewidth=1.2, label="VOL MA(10)")

    ts_to_pos = {ts: idx for idx, ts in enumerate(times)}
    y_span = float(np.nanmax(highs) - np.nanmin(lows))

    for sig in signals:
        if sig.timestamp not in ts_to_pos:
            continue
        pos = ts_to_pos[sig.timestamp]
        is_bottom = sig.signal_type == "BOTTOM_BREAKOUT"
        is_coiled = sig.signal_type == "COILED_MA200_STAGE2"
        badge_color = "#d50000" if is_bottom else ("#6a1b9a" if is_coiled else "#006064")
        tag_title = (
            "★ 引擎A：底部爆量起漲"
            if is_bottom
            else ("★ 引擎B：牛來黏帶穿MA200二拍" if is_coiled else "▲ 引擎A：回踩MA25二次起漲")
        )

        ax_price.scatter(
            [pos],
            [lows[pos] * 0.97],
            marker="^",
            s=180,
            color=badge_color,
            edgecolors="white",
            linewidths=1.2,
            zorder=8,
        )

        extra_line = f"量能: {sig.vol_ratio:.1f}x | 最高漲幅: +{sig.max_gain_after_pct:.1f}% ({sig.max_r_after:.1f}R)"
        if exit_info and sig.timestamp in exit_info:
            ex = exit_info[sig.timestamp]
            extra_line = (
                f"出場: {ex['exit_price']:.6g} ({ex['reason']}) | "
                f"實戰淨利: {ex['pnl_pct']:+.1f}% ({ex['r']:+.1f}R)"
            )
            ex_ts = ex.get("exit_time")
            if ex_ts in ts_to_pos:
                ex_pos = ts_to_pos[ex_ts]
                ax_price.scatter(
                    [ex_pos],
                    [highs[ex_pos] * 1.02],
                    marker="v",
                    s=150,
                    color="#212121",
                    edgecolors="white",
                    linewidths=1.1,
                    zorder=8,
                )

        annot_text = (
            f"{tag_title}\n"
            f"時間: {sig.timestamp.strftime('%m-%d %H:%M')}\n"
            f"進場: {sig.entry_price:.6g} | 止損: {sig.stop_price:.6g}\n"
            f"{extra_line}"
        )
        y_offset = lows[pos] + (0.18 * y_span if is_bottom else 0.24 * y_span)
        ax_price.annotate(
            annot_text,
            xy=(pos, highs[pos]),
            xytext=(max(2, pos - 22), min(float(np.nanmax(highs)) * 0.94, y_offset)),
            fontsize=9.2,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": badge_color, "alpha": 0.92, "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": badge_color, "lw": 1.6},
            zorder=9,
        )

    ax_price.set_title(
        f"{symbol} ({interval}) 起漲點自動偵測圖 v2 — 底部糾結爆量突破 & MA25 趨勢追蹤",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax_price.set_ylabel("價格 (USDT)", fontsize=10)
    ax_price.grid(True, linestyle="--", alpha=0.25)
    ax_price.legend(loc="upper left", ncol=4, fontsize=8.5, framealpha=0.85)

    ax_vol.set_ylabel("成交量", fontsize=10)
    ax_vol.grid(True, linestyle="--", alpha=0.25)
    ax_vol.legend(loc="upper left", ncol=2, fontsize=8.5, framealpha=0.85)

    tick_step = max(1, len(plot_df) // 10)
    tick_indices = list(range(0, len(plot_df), tick_step))
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels([times[i].strftime("%m-%d\n%H:%M") for i in tick_indices], fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_seen_state(path: Path) -> Set[Tuple[str, str]]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {(str(item[0]), str(item[1])) for item in data if isinstance(item, (list, tuple)) and len(item) >= 2}
    except Exception as exc:
        logger.warning("讀取盯盤已通知紀錄失敗 (%s): %s", path, exc)
    return set()


def save_seen_state(path: Path, seen: Set[Tuple[str, str]], max_items: int = 5000) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(seen, key=lambda kv: (kv[1], kv[0]))[-max_items:]
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("寫入盯盤已通知紀錄失敗 (%s): %s", path, exc)


def scan_market_once(
    symbols: Sequence[str],
    interval: str = "15m",
    recent_bars: int = 8,
    workers: int = 8,
    plot_dir: Optional[Path] = None,
    drop_forming: bool = True,
) -> List[BreakoutSignal]:
    """平行掃描多檔幣種，回傳在最近 recent_bars 根已收盤 K 線內觸發的起漲訊號。"""
    found: List[BreakoutSignal] = []

    def _worker(sym: str) -> List[BreakoutSignal]:
        try:
            df = fetch_klines(sym, interval=interval, days=5, drop_forming=drop_forming)
            sigs = detect_signals(df, sym, interval) + detect_coiled_ma200_signals(df, sym, interval)
            if not sigs:
                return []
            if recent_bars > 0:
                min_idx = max(0, len(df) - recent_bars)
                sigs = [s for s in sigs if s.bar_idx >= min_idx]
            if sigs and plot_dir is not None:
                plot_symbol_signals(df, sigs, sym, interval, plot_dir / f"{sym}_{interval}.png")
            return sigs
        except Exception as exc:
            logger.warning("掃描 %s (%s) 失敗: %s", sym, interval, exc)
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            res = fut.result()
            if res:
                found.extend(res)

    found.sort(key=lambda s: (s.timestamp, s.vol_ratio), reverse=True)
    return found


def print_signals_table(signals: Sequence[BreakoutSignal]) -> None:
    if not signals:
        print("目前時間窗口內無符合條件的起漲訊號。")
        return

    print("=" * 120)
    print(
        f"{'時間 (TPE)':<17} | {'幣種':<12} | {'週期':<4} | {'訊號類型':<20} | "
        f"{'進場價':<10} | {'止損價':<10} | {'量能倍數':<8} | {'當根漲幅':<8} | {'後續最高漲幅'}"
    )
    print("-" * 120)
    for s in signals:
        if s.signal_type == "BOTTOM_BREAKOUT":
            stype = "★A1.底部爆量起漲"
        elif s.signal_type == "COILED_MA200_STAGE2":
            stype = "🐂B.牛來黏帶穿MA200"
        else:
            stype = "▲A2.回踩MA25起漲"
        print(
            f"{s.timestamp.strftime('%Y-%m-%d %H:%M'):<17} | {s.symbol:<12} | {s.interval:<4} | {stype:<18} | "
            f"{s.entry_price:<10.6g} | {s.stop_price:<10.6g} | {s.vol_ratio:>6.1f}x  | "
            f"+{s.bar_gain_pct:>5.1f}%  | +{s.max_gain_after_pct:.1f}% ({s.max_r_after:.1f}R)"
        )
        print(f"   └─ 診斷：{s.note}")
    print("=" * 120)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="幣安永續合約起漲點偵測與即時監控腳本 v2")
    parser.add_argument("--symbols", type=str, default="", help="指定幣種，逗號分隔 (例如 BTCUSDT,ETHUSDT)")
    parser.add_argument("--interval", type=str, default="15m", help="K線週期 (5m, 15m, 1h)，預設 15m")
    parser.add_argument("--days", type=int, default=5, help="回溯天數，預設 5 天")
    parser.add_argument("--scan", action="store_true", help="掃描全市場熱門 USDT 永續合約")
    parser.add_argument("--watch", type=int, default=0, help="持續監控秒數 (例如 --watch 60 每60秒掃描一次)")
    parser.add_argument("--top", type=int, default=80, help="全市場掃描的 24h 成交額前 N 名幣種數，預設 80")
    parser.add_argument("--recent-bars", type=int, default=12, help="--scan 模式下只顯示最近 N 根 K 線內的起漲點")
    parser.add_argument("--plot", action="store_true", help="是否輸出標記好起漲點的 K 線圖 PNG")
    parser.add_argument("--include-forming", action="store_true", help="納入尚未收盤的當根 K 線（預設僅掃描已收盤 K 線避免重繪）")
    parser.add_argument("--seen-file", type=str, default="", help="--watch 已推播訊號持久化檔案路徑")
    parser.add_argument("--out-dir", type=str, default="breakout_output", help="圖表輸出資料夾")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    drop_forming = not args.include_forming

    if args.symbols and not args.scan and args.watch <= 0:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        all_signals: List[BreakoutSignal] = []
        for sym in symbols:
            df = fetch_klines(sym, interval=args.interval, days=args.days, drop_forming=drop_forming)
            sigs = detect_signals(df, sym, args.interval) + detect_coiled_ma200_signals(df, sym, args.interval)
            all_signals.extend(sigs)
            if args.plot:
                img_path = out_dir / f"{sym}_{args.interval}.png"
                plot_symbol_signals(df, sigs, sym, args.interval, img_path)
                print(f"[圖表已產生] {img_path.resolve()}")

        print_signals_table(all_signals)
        return 0

    seen_path = Path(args.seen_file) if args.seen_file else (out_dir / ".watch_seen_breakout.json")
    seen: Set[Tuple[str, str]] = load_seen_state(seen_path) if args.watch > 0 else set()
    while True:
        now_str = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now_str} TPE] 正在掃描幣安前 {args.top} 大熱門 USDT 永續合約 ({args.interval} | 已收盤K={drop_forming}) 起漲點...")
        syms = fetch_top_usdt_perps(top_n=args.top)
        sigs = scan_market_once(
            syms,
            interval=args.interval,
            recent_bars=args.recent_bars,
            plot_dir=out_dir if args.plot else None,
            drop_forming=drop_forming,
        )
        if args.watch > 0:
            fresh: List[BreakoutSignal] = []
            for s in sigs:
                key = (s.symbol, s.timestamp.isoformat())
                if key not in seen:
                    seen.add(key)
                    fresh.append(s)
            if fresh:
                save_seen_state(seen_path, seen)
                print_signals_table(fresh)
                print(f"\a[雙引擎起漲提示] 偵測到 {len(fresh)} 筆新已收盤起漲訊號！")
            else:
                print(f"本輪無新增已收盤起漲訊號（視窗內 {len(sigs)} 筆既有訊號已通知過）。")
        else:
            print_signals_table(sigs)
        if args.plot and sigs:
            print(f"已將觸發訊號的 K 線圖存至: {out_dir.resolve()}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
