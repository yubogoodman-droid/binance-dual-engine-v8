#!/usr/bin/env python3
"""幣安永續合約「結構雙引擎二拍突破」即時掃描器（去過擬合・零前視偏差修正版）

設計原則（修正原版 v8 的樣本內湊參數與前視偏差）：
1. 【雙引擎 0% 重疊結構分工（統一二拍確認，拒絕追單根天量插針）】：
   - ★ 引擎 A（水上高台黏帶二拍 STAGE2_SHELF_BREAKOUT）：
     MA200 上揚且連續 >= 80 根穩守水面上 + 7 條均線(MA7~MA120)黏帶 <= 2.0% + 20 根箱體 <= 5.0%
     → 第一拍 2~8x 溫和放量收破箱頂(CLV >= 0.70) + 1~4 根內不破 MA25 且第二拍放量(>=2x, +1%)創收盤新高確認。
   - 🐂 引擎 B（水下築底穿 MA200 二拍 COILED_MA200_STAGE2）：
     MA200 水下連續築底 >= 80 根 + 7 條均線黏帶 <= 2.0% + 長均線走平止跌
     → 第一拍 2~8x 實體紅 K 正式收上 MA200(CLV >= 0.70) + 1~4 根內穩守 MA200 且第二拍放量(>=2x, +1%)確認。
2. 【實盤與回測 100% 對齊】：
   - 預設僅評估「已收線 K 棒」(drop_forming=True)，杜絕盤中未收線假突破誤報。
   - 訊號當下強制檢查「過去 24h 真實滾動成交額 >= 500 萬 USDT」，消除事後熱門幣存活者偏差。
   - 內建 BTCUSDT 大盤急殺保護與同根集體插針過濾，掃描器與回測共用同一套過濾與排序函數。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
BASE = "https://www.binance.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
DEFAULT_MIN_QUOTE_VOL = 5_000_000.0  # 500 萬 USDT 24h 流動性門檻（實盤與回測一致）

_TLS = threading.local()
_RATE_LOCK = threading.Lock()
_PAUSE_UNTIL = 0.0

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
    signal_type: str          # "STAGE2_SHELF_BREAKOUT" | "COILED_MA200_STAGE2"
    bar_idx: int
    timestamp: datetime
    entry_price: float
    stop_price: float
    target_2r: float
    target_4r: float
    vol_ratio: float          # 第一拍突破成交量倍數
    bar_gain_pct: float       # 第二拍確認 K 漲幅 (%)
    ma_spread_pct: float      # 起漲前 7 線黏帶糾結度 (%)
    base_range_pct: float     # 起漲前箱體振幅 / 均線乖離 (%)
    clv: float                # 確認 K 收盤強度 (Close - Low) / (High - Low)
    max_gain_after_pct: float # 訊號後最高漲幅 (僅供圖表觀察，絕不用於過濾或回測)
    max_r_after: float        # 訊號後最高盈虧比 (僅供圖表觀察)
    note: str
    prev_24h_quote_vol: float = 0.0


def _get_session() -> requests.Session:
    sess = getattr(_TLS, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": UA, "Accept": "application/json"})
        _TLS.session = sess
    return sess


def get_json(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 5) -> Any:
    global _PAUSE_UNTIL
    last_err: Optional[Exception] = None
    for i in range(retries):
        wait = _PAUSE_UNTIL - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = _get_session().get(BASE + path, params=params, timeout=20)
            used_weight = int(r.headers.get("x-mbx-used-weight-1m") or 0)
            if used_weight > 2100:
                with _RATE_LOCK:
                    nxt_min = (int(time.time()) // 60 + 1) * 60 + 1.0
                    _PAUSE_UNTIL = max(_PAUSE_UNTIL, nxt_min)
            if r.status_code in (418, 429):
                retry_after = float(r.headers.get("Retry-After") or (1.5 * (i + 1)))
                with _RATE_LOCK:
                    _PAUSE_UNTIL = max(_PAUSE_UNTIL, time.time() + retry_after)
                last_err = RuntimeError(f"HTTP {r.status_code} ({path})")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            time.sleep(0.4 * (i + 1))
    raise RuntimeError(f"API request failed ({path}): {last_err}")


def enrich_indicators(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    """統一計算均線、前 20 根均量倍數與「無前視偏差」前 24 小時滾動成交額。"""
    if df.empty:
        return df
    bar_ms = INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))

    for period, _, _, _ in MA_SPECS:
        df[f"MA{period}"] = df["Close"].rolling(period, min_periods=period).mean()

    df["VOL_MA5"] = df["Volume"].rolling(5, min_periods=3).mean()
    df["VOL_MA10"] = df["Volume"].rolling(10, min_periods=5).mean()
    df["VOL_MA20_PREV"] = df["Volume"].shift(1).rolling(20, min_periods=10).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA20_PREV"].replace(0, np.nan)

    if "QuoteVolume" not in df.columns:
        df["QuoteVolume"] = df["Close"] * df["Volume"]
    df["QUOTEVOL_24H_PREV"] = (
        df["QuoteVolume"]
        .shift(1)
        .rolling(bars_per_day, min_periods=max(1, bars_per_day // 2))
        .sum()
    )
    return df


def fetch_top_usdt_perps(
    top_n: int = 180,
    min_quote_vol: float = DEFAULT_MIN_QUOTE_VOL,
    include_tradfi: bool = False,
) -> List[str]:
    """取得幣安 24h 成交額達標的純加密貨幣 USDT 永續合約清單（不硬塞任何特定幣種）。"""
    info = get_json("/fapi/v1/exchangeInfo")
    valid_syms = set()
    allowed_types = ("PERPETUAL", "TRADIFI_PERPETUAL") if include_tradfi else ("PERPETUAL",)
    for s in info.get("symbols", []):
        sym = s.get("symbol", "")
        if sym in EXCLUDED_SYMBOLS or s.get("underlyingType") == "INDEX":
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


def fetch_klines(
    symbol: str,
    interval: str = "15m",
    days: int = 5,
    drop_forming: bool = True,
) -> pd.DataFrame:
    """拉取幣安永續合約 K 線並計算指標（預設 drop_forming=True 丟棄未收盤 K 線）。"""
    bar_ms = INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = min(1500, max(360, days * bars_per_day + 300))

    raw = get_json(
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": need},
    )
    if not raw:
        return pd.DataFrame()

    now_ms = int(time.time() * 1000)
    if drop_forming and int(raw[-1][0]) + bar_ms > now_ms:
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
    return enrich_indicators(df, interval=interval)


def build_btc_regime_map(btc_df: pd.DataFrame) -> Dict[datetime, bool]:
    """建立零前視偏差的 BTCUSDT 大盤共振濾網：確認 BTC 收盤站穩 MA25 與 MA99 支撐且無 4h 急殺。"""
    if btc_df is None or btc_df.empty or "MA99" not in btc_df.columns or "MA25" not in btc_df.columns:
        return {}
    closes = btc_df["Close"].to_numpy(float)
    ma25 = btc_df["MA25"].to_numpy(float)
    ma99 = btc_df["MA99"].to_numpy(float)
    ts = btc_df.index
    safe_map: Dict[datetime, bool] = {}
    for i in range(len(closes)):
        if i < 16 or not np.isfinite(ma99[i]) or not np.isfinite(ma25[i]) or ma99[i] <= 0 or closes[i - 16] <= 0:
            safe_map[ts[i]] = True
            continue
        chg_16 = (closes[i] / closes[i - 16] - 1.0) * 100.0
        safe_map[ts[i]] = bool(closes[i] >= ma25[i] and closes[i] >= ma99[i] * 0.99 and chg_16 >= -1.5)
    return safe_map


def filter_and_rank_signals(
    signals: Sequence[BreakoutSignal],
    btc_safe_map: Optional[Dict[datetime, bool]] = None,
    max_same_bar_signals: int = 2,
) -> List[BreakoutSignal]:
    """實盤與回測共用的訊號過濾與確定性優先序排列（消除字母順序偏誤與同根集體插針）。"""
    if not signals:
        return []

    btc_filtered = [
        s for s in signals
        if btc_safe_map is None or btc_safe_map.get(s.timestamp, True)
    ]
    ts_counts = Counter(s.timestamp for s in btc_filtered)
    clean = [s for s in btc_filtered if ts_counts[s.timestamp] <= max_same_bar_signals]

    # 同時間觸發時：優先執行結構反轉引擎 B，其次比較均線黏帶緊密度（越小越優先），最後比 24h 成交額
    clean.sort(
        key=lambda s: (
            s.timestamp,
            0 if s.signal_type == "COILED_MA200_STAGE2" else 1,
            s.ma_spread_pct,
            -s.prev_24h_quote_vol,
        )
    )
    return clean


def _min_7ma_spread_pct(m: Dict[int, np.ndarray], start_idx: int, end_idx: int) -> Optional[float]:
    if start_idx < 0 or end_idx <= start_idx:
        return None
    stack = np.vstack([m[p][start_idx:end_idx] for p in (7, 14, 25, 30, 60, 99, 120)])
    if np.isnan(stack).any():
        return None
    mins = np.min(stack, axis=0)
    if np.any(mins <= 0):
        return None
    spreads = (np.max(stack, axis=0) / mins - 1.0) * 100.0
    return float(np.min(spreads))


def detect_signals(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
    *,
    min_bars_above_ma200: int = 80,
    coil_lookback: int = 8,
    max_coil_7_120_pct: float = 2.0,
    box_bars: int = 20,
    max_box_range_pct: float = 5.0,
    max_ma60_over_ma200_pct: float = 4.0,
    min_vol_ratio: float = 2.0,
    max_vol_ratio: float = 8.0,
    min_clv: float = 0.70,
    follow_max_bars: int = 4,
    follow_min_vol_ratio: float = 2.0,
    follow_max_vol_ratio: float = 10.0,
    follow_min_bar_gain_pct: float = 1.0,
    overhead_lookback_bars: int = 60,
    min_quote_vol_24h: float = DEFAULT_MIN_QUOTE_VOL,
    cooldown_bars: int = 24,
) -> List[BreakoutSignal]:
    """引擎 A（水上高台黏帶・二拍突破）：MA200 上揚穩守 >=80 根 + 7 線黏帶 <=2% + 箱體突破二拍確認。"""
    if df.empty or len(df) <= 230:
        return []
    if "VOL_RATIO" not in df.columns or "QUOTEVOL_24H_PREV" not in df.columns:
        df = enrich_indicators(df, interval=interval)

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    vr = df["VOL_RATIO"].to_numpy(float)
    qv24 = df["QUOTEVOL_24H_PREV"].to_numpy(float)
    timestamps = df.index
    m = {p: df[f"MA{p}"].to_numpy(float) for p in (7, 14, 25, 30, 60, 99, 120, 200)}

    n_bars = len(c)
    start_i = max(210, min_bars_above_ma200 + 200, box_bars + 200)
    signals: List[BreakoutSignal] = []
    last_signal_idx = -999

    for i in range(start_i, n_bars - 1):
        if i <= last_signal_idx + cooldown_bars:
            continue
        if not np.isfinite(m[200][i]) or not np.isfinite(m[200][i - 20]) or m[200][i] <= m[200][i - 20]:
            continue
        # 1. 前 80 根收盤全數穩守 MA200 之上（與水下引擎 B 嚴格 0% 重疊）
        if np.any(np.isnan(m[200][i - min_bars_above_ma200 : i])) or np.any(
            c[i - min_bars_above_ma200 : i] < m[200][i - min_bars_above_ma200 : i]
        ):
            continue
        if m[200][i - 1] <= 0 or (m[60][i - 1] / m[200][i - 1] - 1.0) * 100.0 > max_ma60_over_ma200_pct:
            continue

        # 2. 突破前 8 根內 7 條均線黏帶 <= 2.0%，且 20 根箱體振幅 <= 5.0%
        sp7 = _min_7ma_spread_pct(m, i - coil_lookback, i)
        if sp7 is None or sp7 > max_coil_7_120_pct:
            continue

        box_high = float(np.max(h[i - box_bars : i]))
        box_low = float(np.min(l[i - box_bars : i]))
        if box_low <= 0:
            continue
        box_range_pct = (box_high / box_low - 1.0) * 100.0
        if box_range_pct > max_box_range_pct:
            continue

        # 3. 第一拍：實體紅 K 首次收破箱頂，溫和放量 2x~8x（拒絕追 10x~34x 天量針）
        vr_i = vr[i] if np.isfinite(vr[i]) else 0.0
        clv_i = (c[i] - l[i]) / max(h[i] - l[i], 1e-12)
        if not (
            c[i] > box_high
            and c[i - 1] <= box_high
            and c[i] > o[i]
            and min_vol_ratio <= vr_i <= max_vol_ratio
            and clv_i >= min_clv
        ):
            continue

        stop_price = float(min(box_low, m[120][i]) * 0.998)

        # 4. 第二拍確認（1~4 根內不破 MA25，溫和放量 2x~10x 突破前 60 根高點，拒絕追高天量衰竭針）
        overhead_high = float(np.max(h[max(0, i - overhead_lookback_bars) : i]))
        for j in range(i + 1, min(n_bars, i + follow_max_bars + 1)):
            if not np.isfinite(m[25][j]) or c[j] < m[25][j]:
                break
            qv_j = float(qv24[j]) if np.isfinite(qv24[j]) else 0.0
            if min_quote_vol_24h > 0 and qv_j < min_quote_vol_24h:
                continue
            vr_j = float(vr[j]) if np.isfinite(vr[j]) else 0.0
            if not (c[j] > o[j] and c[j] > c[i] and c[j] >= overhead_high and follow_min_vol_ratio <= vr_j <= follow_max_vol_ratio):
                continue
            clv_j = float((c[j] - l[j]) / max(h[j] - l[j], 1e-12))
            gain_j = float((c[j] / max(o[j], 1e-12) - 1.0) * 100.0)
            if clv_j < min_clv or gain_j < follow_min_bar_gain_pct:
                continue
            if not (m[7][j] > m[14][j] > m[25][j]):
                continue

            entry_px = float(c[j])
            risk = entry_px - stop_price
            if risk <= 0:
                break
            future_high = float(np.max(h[j:]))
            max_gain_after = (future_high - entry_px) / entry_px * 100.0
            max_r = (future_high - entry_px) / risk

            signals.append(
                BreakoutSignal(
                    symbol=symbol,
                    interval=interval,
                    signal_type="STAGE2_SHELF_BREAKOUT",
                    bar_idx=j,
                    timestamp=timestamps[j],
                    entry_price=entry_px,
                    stop_price=stop_price,
                    target_2r=entry_px + 2.0 * risk,
                    target_4r=entry_px + 4.0 * risk,
                    vol_ratio=float(vr_i),
                    bar_gain_pct=gain_j,
                    ma_spread_pct=sp7,
                    base_range_pct=box_range_pct,
                    clv=clv_j,
                    max_gain_after_pct=max_gain_after,
                    max_r_after=max_r,
                    note=(
                        f"水上高台(>={min_bars_above_ma200}根守MA200) + 7線黏帶({sp7:.2f}%) + 箱體({box_range_pct:.1f}%) "
                        f"→ 第一拍 {vr_i:.1f}x 破箱頂 + 第二拍(+{gain_j:.1f}%, 量{vr_j:.1f}x, 24h額${qv_j/1e6:.1f}M)確認"
                    ),
                    prev_24h_quote_vol=qv_j,
                )
            )
            last_signal_idx = j
            break

    return signals


def detect_coiled_ma200_signals(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
    *,
    min_bars_below_ma200: int = 80,
    coil_lookback: int = 8,
    max_coil_7_120_pct: float = 2.0,
    slope_lookback: int = 16,
    max_ext_ma200_pct: float = 3.5,
    min_vol_ratio: float = 2.0,
    max_vol_ratio: float = 8.0,
    min_clv: float = 0.70,
    follow_max_bars: int = 4,
    follow_min_vol_ratio: float = 2.0,
    follow_max_vol_ratio: float = 10.0,
    follow_min_bar_gain_pct: float = 1.0,
    min_quote_vol_24h: float = DEFAULT_MIN_QUOTE_VOL,
    cooldown_bars: int = 24,
) -> List[BreakoutSignal]:
    """引擎 B（水下築底黏帶・穿 MA200 二拍）：水下 >=80 根 + 7 線黏帶 <=2% + 穿 MA200 二拍確認。"""
    if df.empty or len(df) <= 230:
        return []
    if "VOL_RATIO" not in df.columns or "QUOTEVOL_24H_PREV" not in df.columns:
        df = enrich_indicators(df, interval=interval)

    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    vr = df["VOL_RATIO"].to_numpy(float)
    qv24 = df["QUOTEVOL_24H_PREV"].to_numpy(float)
    timestamps = df.index
    m = {p: df[f"MA{p}"].to_numpy(float) for p in (7, 14, 25, 30, 60, 99, 120, 200)}

    n_bars = len(c)
    start_i = max(210, min_bars_below_ma200 + 200, slope_lookback + 200)
    signals: List[BreakoutSignal] = []
    last_signal_idx = -999

    for i in range(start_i, n_bars - 1):
        if i <= last_signal_idx + cooldown_bars:
            continue
        m200_i = m[200][i]
        if not np.isfinite(m200_i) or m200_i <= 0 or not np.isfinite(m[200][i - 1]):
            continue

        # 1. 第一拍正式從下方收盤突破 MA200，且前 >= 80 根全數位於 MA200 水下築底
        if not (c[i] > m200_i and c[i - 1] <= m[200][i - 1]):
            continue
        if np.any(np.isnan(m[200][i - min_bars_below_ma200 : i])) or np.any(
            c[i - min_bars_below_ma200 : i] > m[200][i - min_bars_below_ma200 : i]
        ):
            continue

        bars_below = 0
        for k in range(i - 1, max(0, i - 201), -1):
            if np.isnan(m[200][k]) or c[k] > m[200][k]:
                break
            bars_below += 1

        # 2. 突破前 8 根內 7 條均線黏帶 <= 2.0%，且長均線走平止跌
        sp7 = _min_7ma_spread_pct(m, i - coil_lookback, i)
        if sp7 is None or sp7 > max_coil_7_120_pct:
            continue
        if not (m[60][i] >= m[60][i - slope_lookback] and m200_i >= m[200][i - slope_lookback] * 0.994):
            continue

        # 3. 第一拍：2x~8x 溫和放量紅 K，收盤飽滿 CLV >= 0.70，不過度追高乖離 (<= 3.5%)
        vr_i = vr[i] if np.isfinite(vr[i]) else 0.0
        clv_i = (c[i] - l[i]) / max(h[i] - l[i], 1e-12)
        ext_ma200_pct = (c[i] / m200_i - 1.0) * 100.0
        if not (
            c[i] > o[i]
            and min_vol_ratio <= vr_i <= max_vol_ratio
            and clv_i >= min_clv
            and ext_ma200_pct <= max_ext_ma200_pct
        ):
            continue

        stop_price = float(min(m[60][i], m[99][i], m[120][i], float(np.min(l[max(0, i - 3) : i + 1]))))

        # 4. 第二拍確認（1~4 根內穩守 MA200 之上，2x~10x 溫和放量紅 K >=1% 確認，過濾 >10x 天量頂部爆衝）
        for j in range(i + 1, min(n_bars, i + follow_max_bars + 1)):
            if not np.isfinite(m[200][j]) or c[j] <= m[200][j]:
                break
            qv_j = float(qv24[j]) if np.isfinite(qv24[j]) else 0.0
            if min_quote_vol_24h > 0 and qv_j < min_quote_vol_24h:
                continue
            vr_j = float(vr[j]) if np.isfinite(vr[j]) else 0.0
            if not (c[j] > o[j] and c[j] > c[i] and follow_min_vol_ratio <= vr_j <= follow_max_vol_ratio):
                continue
            clv_j = float((c[j] - l[j]) / max(h[j] - l[j], 1e-12))
            gain_j = float((c[j] / max(o[j], 1e-12) - 1.0) * 100.0)
            if clv_j < min_clv or gain_j < follow_min_bar_gain_pct:
                continue
            if not (m[7][j] > m[14][j] > m[25][j]):
                continue

            entry_px = float(c[j])
            risk = entry_px - stop_price
            if risk <= 0:
                break
            future_high = float(np.max(h[j:]))
            max_gain_after = (future_high - entry_px) / entry_px * 100.0
            max_r = (future_high - entry_px) / risk

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
                    vol_ratio=float(vr_i),
                    bar_gain_pct=gain_j,
                    ma_spread_pct=sp7,
                    base_range_pct=ext_ma200_pct,
                    clv=clv_j,
                    max_gain_after_pct=max_gain_after,
                    max_r_after=max_r,
                    note=(
                        f"水下築底({bars_below}根) + 7線黏帶({sp7:.2f}%) "
                        f"→ 第一拍 {vr_i:.1f}x 穿MA200 + 第二拍(+{gain_j:.1f}%, 量{vr_j:.1f}x, 24h額${qv_j/1e6:.1f}M)確認"
                    ),
                    prev_24h_quote_vol=qv_j,
                )
            )
            last_signal_idx = j
            break

    return signals


def plot_symbol_signals(
    df: pd.DataFrame,
    signals: List[BreakoutSignal],
    symbol: str,
    interval: str,
    out_path: Path,
    exit_info: Optional[Dict[datetime, Dict[str, Any]]] = None,
) -> None:
    """繪製仿幣安風格的 K 線 + 8 條均線 + 成交量 + 二拍起漲點(及出場點)標記圖。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "WenQuanYi Micro Hei", "Noto Sans CJK TC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
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
        is_coiled = sig.signal_type == "COILED_MA200_STAGE2"
        badge_color = "#6a1b9a" if is_coiled else "#0277bd"
        tag_title = "🐂 引擎B：水下築底穿MA200二拍" if is_coiled else "★ 引擎A：水上高台黏帶二拍"

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

        extra_line = f"首拍量: {sig.vol_ratio:.1f}x | 24h額: ${sig.prev_24h_quote_vol/1e6:.1f}M"
        if exit_info and sig.timestamp in exit_info:
            ex = exit_info[sig.timestamp]
            extra_line = (
                f"出場: {ex['exit_price']:.6g} ({ex['reason']}) | "
                f"實戰淨利: {ex['pnl_pct']:+.1f}% ({ex['r']:+.2f}R)"
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
        y_offset = lows[pos] + (0.22 * y_span if is_coiled else 0.18 * y_span)
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
        f"{symbol} ({interval}) 結構雙引擎二拍確認圖（已排除未收盤K線與低流動性雜訊）",
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


def scan_market_once(
    symbols: Sequence[str],
    interval: str = "15m",
    recent_bars: int = 8,
    workers: int = 8,
    plot_dir: Optional[Path] = None,
    min_quote_vol: float = DEFAULT_MIN_QUOTE_VOL,
) -> List[BreakoutSignal]:
    """平行掃描多檔幣種，套用與回測完全一致的已收線 K 棒、BTC 急殺保護與同根集體插針過濾。"""
    raw_found: List[BreakoutSignal] = []
    dfs_with_hits: Dict[str, pd.DataFrame] = {}

    try:
        btc_df = fetch_klines("BTCUSDT", interval=interval, days=5, drop_forming=True)
        btc_safe_map = build_btc_regime_map(btc_df)
    except Exception:
        btc_safe_map = {}

    def _worker(sym: str) -> tuple[str, pd.DataFrame, List[BreakoutSignal]]:
        try:
            df = fetch_klines(sym, interval=interval, days=5, drop_forming=True)
            sigs = (
                detect_signals(df, sym, interval, min_quote_vol_24h=min_quote_vol)
                + detect_coiled_ma200_signals(df, sym, interval, min_quote_vol_24h=min_quote_vol)
            )
            if not sigs:
                return sym, df, []
            if recent_bars > 0:
                min_idx = max(0, len(df) - recent_bars)
                sigs = [s for s in sigs if s.bar_idx >= min_idx]
            return sym, df, sigs
        except Exception:
            return sym, pd.DataFrame(), []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym, df, res = fut.result()
            if res:
                raw_found.extend(res)
                dfs_with_hits[sym] = df

    filtered = filter_and_rank_signals(raw_found, btc_safe_map=btc_safe_map, max_same_bar_signals=2)
    if plot_dir is not None and filtered:
        by_sym: Dict[str, List[BreakoutSignal]] = {}
        for s in filtered:
            by_sym.setdefault(s.symbol, []).append(s)
        for sym, sigs in by_sym.items():
            if sym in dfs_with_hits:
                plot_symbol_signals(dfs_with_hits[sym], sigs, sym, interval, plot_dir / f"{sym}_{interval}.png")

    return list(reversed(filtered))


def print_signals_table(signals: Sequence[BreakoutSignal]) -> None:
    if not signals:
        print("目前已收盤 K 線時間窗口內無符合條件的二拍確認訊號。")
        return

    print("=" * 124)
    print(
        f"{'收盤時間 (TPE)':<17} | {'幣種':<12} | {'週期':<4} | {'引擎類型':<20} | "
        f"{'進場價':<10} | {'止損價':<10} | {'首拍量能':<8} | {'24h成交額':<10}"
    )
    print("-" * 124)
    for s in signals:
        stype = "🐂B.水下穿MA200二拍" if s.signal_type == "COILED_MA200_STAGE2" else "★A.水上高台二拍"
        print(
            f"{s.timestamp.strftime('%Y-%m-%d %H:%M'):<17} | {s.symbol:<12} | {s.interval:<4} | {stype:<18} | "
            f"{s.entry_price:<10.6g} | {s.stop_price:<10.6g} | {s.vol_ratio:>6.1f}x  | "
            f"${s.prev_24h_quote_vol/1e6:>7.1f}M"
        )
        print(f"   └─ 診斷：{s.note}")
    print("=" * 124)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="幣安永續合約結構雙引擎二拍突破掃描器（去過擬合修正版）")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT", help="指定幣種，逗號分隔")
    parser.add_argument("--interval", type=str, default="15m", help="K線週期 (5m, 15m, 1h)，預設 15m")
    parser.add_argument("--days", type=int, default=5, help="回溯天數，預設 5 天")
    parser.add_argument("--scan", action="store_true", help="掃描全市場流動性達標之 USDT 永續合約")
    parser.add_argument("--watch", type=int, default=0, help="持續監控秒數 (例如 --watch 60 每60秒掃描一次)")
    parser.add_argument("--top", type=int, default=180, help="全市場掃描的 24h 成交額前 N 名幣種數，預設 180")
    parser.add_argument("--min-qv", type=float, default=DEFAULT_MIN_QUOTE_VOL, help="最低 24h 成交額門檻 (USDT)，預設 5,000,000")
    parser.add_argument("--recent-bars", type=int, default=12, help="--scan 模式下只顯示最近 N 根已收盤 K 線內的訊號")
    parser.add_argument("--include-forming", action="store_true", help="允許評估尚未收盤的當根 K 線（預設關閉以防假突破）")
    parser.add_argument("--plot", action="store_true", help="是否輸出標記好起漲點的 K 線圖 PNG")
    parser.add_argument("--out-dir", type=str, default="breakout_output", help="圖表輸出資料夾")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.scan or args.watch > 0:
        while True:
            now_str = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{now_str} TPE] 正在掃描幣安前 {args.top} 大流動性達標 (>=${args.min_qv/1e6:.0f}M) "
                f"USDT 永續合約 ({args.interval}, 已收盤K線)..."
            )
            syms = fetch_top_usdt_perps(top_n=args.top, min_quote_vol=args.min_qv)
            sigs = scan_market_once(
                syms,
                interval=args.interval,
                recent_bars=args.recent_bars,
                plot_dir=out_dir if args.plot else None,
                min_quote_vol=args.min_qv,
            )
            print_signals_table(sigs)
            if args.plot and sigs:
                print(f"已將觸發訊號的 K 線圖存至: {out_dir.resolve()}")
            if args.watch <= 0:
                break
            time.sleep(args.watch)
        return 0

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    drop_forming = not args.include_forming
    try:
        btc_df = fetch_klines("BTCUSDT", interval=args.interval, days=args.days, drop_forming=drop_forming)
        btc_safe_map = build_btc_regime_map(btc_df)
    except Exception:
        btc_safe_map = {}

    all_signals: List[BreakoutSignal] = []
    for sym in symbols:
        df = fetch_klines(sym, interval=args.interval, days=args.days, drop_forming=drop_forming)
        sigs = (
            detect_signals(df, sym, args.interval, min_quote_vol_24h=args.min_qv)
            + detect_coiled_ma200_signals(df, sym, args.interval, min_quote_vol_24h=args.min_qv)
        )
        sigs = filter_and_rank_signals(sigs, btc_safe_map=btc_safe_map, max_same_bar_signals=99)
        all_signals.extend(sigs)
        if args.plot and not df.empty:
            img_path = out_dir / f"{sym}_{args.interval}.png"
            plot_symbol_signals(df, sigs, sym, args.interval, img_path)
            print(f"[圖表已產生] {img_path.resolve()}")

    print_signals_table(all_signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
