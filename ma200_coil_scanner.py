#!/usr/bin/env python3
"""幣安永續合約「MA200 附近均線粘合起漲」偵測與即時掃描腳本

專抓類似 FILUSDT 15m（MA7~MA200 高度糾結粘合、水下築底後帶量突破 MA200）的起漲點：
1. 長期水下築底：突破 MA200 前 24h (96 根) 至少 85% 收盤價位於 MA200 下方或貼邊。
2. 8 線極致粘合：MA7, MA14, MA25, MA30, MA60, MA99, MA120, MA200 全 8 條均線糾結度 <= 1.50%，
   且長期均線簇 (MA99, MA120, MA200) 糾結度 <= 1.20%、12h 箱體振幅 <= 4.50%。
3. 護盤與趨勢確認：起漲前連 8 根至少 7 根收盤站穩 12h VWAP，MA60 翻揚 (>= +0.25%)，MA200 走平 (>= -0.30%)。
4. 點火位置精準在 MA200 附近：現價位於 MA200 上方 +0.20% ~ +3.00%（不追高乖離），
   並伴隨 >= 3.8x 成交額放量、>= 55% 主動買盤與 >= 0.70 K 線收盤強度 (CLV)。

常用指令：
  # 1. 掃描全市場即時起漲點
  python ma200_coil_scanner.py --scan --plot

  # 2. 每 60 秒持續盯盤（新訊號響鈴提示）
  python ma200_coil_scanner.py --watch 60 --plot

  # 3. 檢查指定幣種與歷史日期（例如重現 FILUSDT 2026-01-01 起漲點）
  python ma200_coil_scanner.py --symbols FILUSDT --end-date 2026-01-02 --days 6 --plot

  # 4. 使用本地快取執行全市場回測驗證
  python ma200_coil_scanner.py --backtest-cache breakout_output/cache_365d
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
BINANCE_FAPI = "https://fapi.binance.com"
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

INTERVAL_MS = {
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
}

# 對齊手機幣安 App 均線顏色
MA_SPECS = [
    (7, "#f0b90b", 1.2, "MA(7)"),
    (14, "#26c6da", 1.1, "MA(14)"),
    (25, "#ec407a", 1.5, "MA(25)"),
    (30, "#2962ff", 1.0, "MA(30)"),
    (60, "#2e7d32", 1.1, "MA(60)"),
    (99, "#7e57c2", 1.3, "MA(99)"),
    (120, "#66bb6a", 1.2, "MA(120)"),
    (200, "#8e24aa", 1.8, "MA(200)"),
]

EXCLUDED_SYMBOLS = frozenset({
    "USDCUSDT", "FDUSDUSDT", "USDEUSDT", "TUSDUSDT",
    "SQQQUSDT", "SOXSUSDT", "SKDDUSDT", "TBTUSDT", "TZAUSDT", "UVXYUSDT",
    "TQQQUSDT", "SOXLUSDT", "SPCXUSDT", "QQQUSDT", "SPYUSDT",
    "NVDAUSDT", "TSLAUSDT", "AAPLUSDT", "AMDUSDT", "MSTRUSDT", "COINUSDT",
    "GOOGLUSDT", "METAUSDT", "AMZNUSDT", "MSFTUSDT", "PLTRUSDT", "HOODUSDT",
    "SMCIUSDT", "AVGOUSDT", "MUUSDT", "MUUUSDT", "BABAUSDT", "NBISUSDT",
    "ONDSUSDT", "CRCLUSDT", "SNDKUSDT", "SNXXUSDT", "SKHYNIXUSDT", "SKHYUSDT",
    "KORUUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "BZUSDT",
})

SNIPER_PARAMS: Dict[str, float] = {
    "min_below96_ratio": 0.85,
    "max_spread8_pct": 1.50,
    "max_spread_long_pct": 1.20,
    "max_rng48_pct": 4.50,
    "min_abv_vwap": 5,
    "min_qvm": 3.50,
    "min_vol_acc": 1.00,
    "min_quote_vol": 100_000.0,
    "min_tbr": 0.55,
    "min_clv": 0.70,
    "min_gain_pct": 0.45,
    "max_rsi6": 90.0,
    "min_ext200_pct": 0.20,
    "max_ext200_pct": 3.00,
    "min_slp60_pct": 0.25,
    "min_slp99_pct": -0.50,
    "min_slp200_pct": -0.30,
    "max_cross_lag": 6,
}

BALANCED_PARAMS: Dict[str, float] = {
    "min_below96_ratio": 0.85,
    "max_spread8_pct": 1.50,
    "max_spread_long_pct": 1.20,
    "max_rng48_pct": 4.50,
    "min_abv_vwap": 5,
    "min_qvm": 3.50,
    "min_vol_acc": 1.00,
    "min_quote_vol": 100_000.0,
    "min_tbr": 0.55,
    "min_clv": 0.70,
    "min_gain_pct": 0.45,
    "max_rsi6": 90.0,
    "min_ext200_pct": 0.20,
    "max_ext200_pct": 3.00,
    "min_slp60_pct": 0.15,
    "min_slp99_pct": -0.50,
    "min_slp200_pct": -0.30,
    "max_cross_lag": 6,
}

RELAXED_PARAMS: Dict[str, float] = {
    "min_below96_ratio": 0.75,
    "max_spread8_pct": 2.20,
    "max_spread_long_pct": 1.80,
    "max_rng48_pct": 6.00,
    "min_abv_vwap": 5,
    "min_qvm": 2.50,
    "min_vol_acc": 1.00,
    "min_quote_vol": 100_000.0,
    "min_tbr": 0.53,
    "min_clv": 0.65,
    "min_gain_pct": 0.35,
    "max_rsi6": 95.0,
    "min_ext200_pct": 0.15,
    "max_ext200_pct": 3.50,
    "min_slp60_pct": 0.10,
    "min_slp99_pct": -0.60,
    "min_slp200_pct": -0.45,
    "max_cross_lag": 8,
}

PRESETS: Dict[str, Dict[str, float]] = {
    "sniper": SNIPER_PARAMS,
    "balanced": BALANCED_PARAMS,
    "relaxed": RELAXED_PARAMS,
}


@dataclass
class MA200CoilSignal:
    symbol: str
    interval: str
    bar_idx: int
    cross_idx: int
    timestamp: datetime
    entry_price: float
    stop_price: float
    ma200: float
    ext200_pct: float
    spread8_pct: float
    spread_long_pct: float
    rng48_pct: float
    qvm: float
    vol_acc: float
    vol_ratio: float
    taker_buy_pct: float
    clv: float
    rsi6: float
    bar_gain_pct: float
    below96_pct: float
    risk_pct: float
    score: int
    max_gain_after_pct: float
    note: str


def api_get(path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Any:
    """呼叫幣安永續合約 API 並自動處理限流重試。"""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = HTTP.get(BINANCE_FAPI + path, params=params, timeout=15)
            if resp.status_code in (418, 429):
                time.sleep(1.5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Binance API failed ({path}): {last_err}")


def list_usdt_perps(min_quote_vol: float = 1_000_000.0, top_n: int = 220) -> List[str]:
    """列出 24h 成交額前 N 名且非黑名單的純加密 USDT 永續合約。"""
    info = api_get("/fapi/v1/exchangeInfo")
    tradable: Set[str] = {
        item["symbol"]
        for item in info.get("symbols", [])
        if item.get("quoteAsset") == "USDT"
        and item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("symbol") not in EXCLUDED_SYMBOLS
    }
    tickers = api_get("/fapi/v1/ticker/24hr")
    ranked: List[Tuple[float, str]] = []
    for row in tickers:
        sym = row.get("symbol", "")
        if sym not in tradable:
            continue
        qv = float(row.get("quoteVolume") or 0.0)
        if qv >= min_quote_vol:
            ranked.append((qv, sym))
    ranked.sort(reverse=True)
    return [sym for _, sym in ranked[:top_n]]


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """使用平滑 RSI (Wilder/EMA) 計算與手機幣安一致的 RSI 數值。"""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算 MA7~MA200、VOL 均線、成交額倍數、主動買盤佔比、12h VWAP 與 RSI。"""
    out = df.copy()
    for period, _, _, _ in MA_SPECS:
        out[f"MA{period}"] = out["Close"].rolling(period, min_periods=period).mean()

    out["VOL_MA5"] = out["Volume"].rolling(5, min_periods=3).mean()
    out["VOL_MA10"] = out["Volume"].rolling(10, min_periods=5).mean()
    vol_ma20_prev = out["Volume"].shift(1).rolling(20, min_periods=10).mean()
    out["VOL_RATIO"] = out["Volume"] / vol_ma20_prev.replace(0.0, np.nan)

    qv_med_prev = out["QuoteVol"].shift(1).rolling(20, min_periods=10).median()
    out["QVM"] = out["QuoteVol"] / qv_med_prev.replace(0.0, np.nan)
    out["TBR"] = out["TakerBuyVol"] / out["Volume"].replace(0.0, np.nan)

    vol48 = out["Volume"].rolling(48, min_periods=24).sum()
    qv48 = out["QuoteVol"].rolling(48, min_periods=24).sum()
    out["VWAP12"] = np.where(vol48 > 0, qv48 / vol48, np.nan)

    for p in (6, 12, 24):
        out[f"RSI{p}"] = compute_rsi(out["Close"], p)
    return out


def fetch_klines(
    symbol: str,
    interval: str = "15m",
    days: int = 5,
    end_dt: Optional[datetime] = None,
    drop_forming: bool = False,
) -> pd.DataFrame:
    """分頁拉取指定幣種與區間的幣安永續 K 線資料（含主動買盤欄位）。"""
    bar_ms = INTERVAL_MS.get(interval, 900_000)
    bars_per_day = max(1, int(86_400_000 // bar_ms))
    need = days * bars_per_day + 260

    end_ms = int(end_dt.timestamp() * 1000) if end_dt is not None else None
    now_ms = int(time.time() * 1000)
    raw: list = []

    while len(raw) < need:
        limit = min(1000, need - len(raw))
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        chunk = api_get("/fapi/v1/klines", params=params)
        if not chunk:
            break
        raw = list(chunk) + raw
        end_ms = int(chunk[0][0]) - 1
        if len(chunk) < limit:
            break
        time.sleep(0.03)

    if not raw:
        return pd.DataFrame()
    if drop_forming and int(raw[-1][0]) + bar_ms > now_ms:
        raw = raw[:-1]

    df = pd.DataFrame(
        {
            "Open": [float(x[1]) for x in raw],
            "High": [float(x[2]) for x in raw],
            "Low": [float(x[3]) for x in raw],
            "Close": [float(x[4]) for x in raw],
            "Volume": [float(x[5]) for x in raw],
            "QuoteVol": [float(x[7]) for x in raw],
            "TakerBuyVol": [float(x[9]) for x in raw],
        },
        index=pd.to_datetime([int(x[0]) for x in raw], unit="ms", utc=True).tz_convert(TPE),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return compute_indicators(df)


def detect_ma200_coil_breakouts(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
    *,
    preset: str = "sniper",
    relaxed: bool = False,
    cooldown_bars: int = 16,
) -> List[MA200CoilSignal]:
    """偵測位於 MA200 附近的『水下築底 + 8線粘合 + VWAP護盤 + 主動量能加速起漲點』。"""
    if df.empty or len(df) < 250:
        return []

    active_preset = "relaxed" if relaxed else preset
    cfg = PRESETS.get(active_preset, SNIPER_PARAMS)
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    qv = df["QuoteVol"].to_numpy(float)
    vr = df["VOL_RATIO"].to_numpy(float)
    qvm = df["QVM"].to_numpy(float)
    tbr = df["TBR"].to_numpy(float)
    vwap12 = df["VWAP12"].to_numpy(float)
    rsi6 = df["RSI6"].to_numpy(float)
    m7 = df["MA7"].to_numpy(float)
    m14 = df["MA14"].to_numpy(float)
    m25 = df["MA25"].to_numpy(float)
    m30 = df["MA30"].to_numpy(float)
    m60 = df["MA60"].to_numpy(float)
    m99 = df["MA99"].to_numpy(float)
    m120 = df["MA120"].to_numpy(float)
    m200 = df["MA200"].to_numpy(float)
    times = df.index

    signals: List[MA200CoilSignal] = []
    last_sig = -999
    max_lag = int(cfg["max_cross_lag"])

    with np.errstate(invalid="ignore", divide="ignore"):
        cand_mask = (
            (c > o)
            & (o > 0)
            & (h > l)
            & (qvm >= cfg["min_qvm"])
            & (qv >= cfg["min_quote_vol"])
            & (tbr >= cfg["min_tbr"])
            & (rsi6 <= cfg["max_rsi6"])
            & (c > m7)
            & (m7 > m14)
            & (m14 > m25)
            & (c > vwap12)
        )
    cand_mask[:250] = False

    for raw_i in np.flatnonzero(cand_mask):
        i = int(raw_i)
        if i - last_sig < cooldown_bars:
            continue

        needed = (
            m7[i], m14[i], m25[i], m30[i], m60[i], m99[i], m120[i], m200[i],
            m60[i - 24], m99[i - 24], m200[i - 24], qvm[i], tbr[i], vwap12[i], rsi6[i],
        )
        if not np.isfinite(needed).all():
            continue

        # 1. 價位緊貼 MA200 上緣起漲，不追高乖離
        ext200 = (c[i] / m200[i] - 1.0) * 100.0
        if not (cfg["min_ext200_pct"] <= ext200 <= cfg["max_ext200_pct"]):
            continue

        # 2. 剛由下方穿越 MA200（當根或前 1~max_lag 根內），且站穩不破
        cross_bars = [
            k for k in range(max(1, i - max_lag), i + 1)
            if np.isfinite(m200[k]) and np.isfinite(m200[k - 1]) and c[k] > m200[k] and c[k - 1] <= m200[k - 1]
        ]
        if not cross_bars:
            continue
        cross_idx = cross_bars[-1]
        if cross_idx < 200 or np.min(c[cross_idx : i + 1] / m200[cross_idx : i + 1]) < 0.998:
            continue

        # 3. 穿越前 24h (96根) 至少 85% 收盤於 MA200 水下（真築底，非高位反抽）
        below96 = float(np.mean(c[cross_idx - 96 : cross_idx] <= m200[cross_idx - 96 : cross_idx]))
        if below96 < cfg["min_below96_ratio"]:
            continue

        # 4. 穿越當下 8 線高度粘合 + 長期均線簇走平糾結 + 12h 窄幅箱體
        ma8_cross = (
            m7[cross_idx], m14[cross_idx], m25[cross_idx], m30[cross_idx],
            m60[cross_idx], m99[cross_idx], m120[cross_idx], m200[cross_idx],
        )
        if not np.isfinite(ma8_cross).all():
            continue
        spread8 = (max(ma8_cross) / min(ma8_cross) - 1.0) * 100.0
        spread_long = (max(m99[i], m120[i], m200[i]) / min(m99[i], m120[i], m200[i]) - 1.0) * 100.0
        if spread8 > cfg["max_spread8_pct"] or spread_long > cfg["max_spread_long_pct"]:
            continue

        lo48 = float(np.min(l[cross_idx - 48 : cross_idx]))
        hi48 = float(np.max(h[cross_idx - 48 : cross_idx]))
        if lo48 <= 0:
            continue
        rng48 = (hi48 / lo48 - 1.0) * 100.0
        if rng48 > cfg["max_rng48_pct"]:
            continue

        # 5. MA60 翻揚 + MA99/MA200 走平不再下壓 + 前 8 根連站 12h VWAP 護盤
        slp60 = (m60[i] / m60[i - 24] - 1.0) * 100.0
        slp99 = (m99[i] / m99[i - 24] - 1.0) * 100.0
        slp200 = (m200[i] / m200[i - 24] - 1.0) * 100.0
        if (
            slp60 < cfg["min_slp60_pct"]
            or slp99 < cfg.get("min_slp99_pct", -99.0)
            or slp200 < cfg["min_slp200_pct"]
        ):
            continue

        abv_vwap = int(np.sum(c[i - 8 : i] >= vwap12[i - 8 : i]))
        if abv_vwap < cfg["min_abv_vwap"] or c[i] <= vwap12[i]:
            continue

        # 6. 短線均線多頭排列 + 排除前 6 根已有插針出貨
        if not (m7[i] > m14[i] > m25[i] and c[i] > m7[i]):
            continue
        had_wick = any(
            l[k] > 0 and (h[k] - l[k]) / l[k] * 100.0 > 4.0 and qvm[k] > 6.0
            for k in range(max(0, i - 6), i)
        )
        if had_wick:
            continue

        # 7. 量能加速爆發 + 主動買盤 + 飽滿實體收盤 + 排除 RSI(6) 過熱追高
        clv = (c[i] - l[i]) / (h[i] - l[i])
        gain = (c[i] - o[i]) / o[i] * 100.0
        vol_acc = float(qv[i] / max(qv[i - 1], 1.0))
        if (
            qvm[i] < cfg["min_qvm"]
            or vol_acc < cfg["min_vol_acc"]
            or qv[i] < cfg["min_quote_vol"]
            or tbr[i] < cfg["min_tbr"]
            or clv < cfg["min_clv"]
            or gain < cfg["min_gain_pct"]
            or rsi6[i] > cfg["max_rsi6"]
        ):
            continue

        # 8. 止損設在 MA200 粘合底部下方，限制合理風險空間
        base_low = float(np.min(l[max(0, cross_idx - 6) : i + 1]))
        stop_price = min(base_low, m200[i] * 0.992) * 0.996
        risk_pct = (c[i] - stop_price) / c[i] * 100.0
        if not (0.80 <= risk_pct <= 7.00):
            continue

        future_high = float(np.max(h[i : min(len(df), i + 289)]))
        max_gain_after = (future_high / c[i] - 1.0) * 100.0

        # 綜合品質評分 (0~99)
        s_coil = max(0.0, min(30.0, (2.2 - spread8) / 1.7 * 30.0))
        s_vol = max(0.0, min(25.0, (min(qvm[i], 10.0) - 2.0) / 8.0 * 25.0))
        s_tbr = max(0.0, min(25.0, (tbr[i] - 0.50) / 0.16 * 25.0))
        s_clv = max(0.0, min(20.0, (clv - 0.60) / 0.40 * 20.0))
        bonus = 12 if active_preset != "relaxed" else 0
        score = min(99, int(round(s_coil + s_vol + s_tbr + s_clv + bonus)))

        stage_tag = "初穿MA200點火" if i == cross_idx else f"站穩MA200第{i - cross_idx}拍確認"
        note = (
            f"{stage_tag} | 水下{below96*100:.0f}% + 8線黏合({spread8:.2f}%) + 12h振幅({rng48:.2f}%) "
            f"→ 放量{qvm[i]:.1f}x(加速{vol_acc:.1f}x) + 主買{tbr[i]*100:.1f}% + RSI6({rsi6[i]:.1f}) + 距MA200({ext200:+.2f}%)"
        )

        last_sig = i
        signals.append(
            MA200CoilSignal(
                symbol=symbol,
                interval=interval,
                bar_idx=i,
                cross_idx=cross_idx,
                timestamp=times[i],
                entry_price=float(c[i]),
                stop_price=float(stop_price),
                ma200=float(m200[i]),
                ext200_pct=float(ext200),
                spread8_pct=float(spread8),
                spread_long_pct=float(spread_long),
                rng48_pct=float(rng48),
                qvm=float(qvm[i]),
                vol_acc=vol_acc,
                vol_ratio=float(vr[i]) if np.isfinite(vr[i]) else float(qvm[i]),
                taker_buy_pct=float(tbr[i] * 100.0),
                clv=float(clv),
                rsi6=float(rsi6[i]),
                bar_gain_pct=float(gain),
                below96_pct=float(below96 * 100.0),
                risk_pct=float(risk_pct),
                score=score,
                max_gain_after_pct=float(max_gain_after),
                note=note,
            )
        )

    return signals


def simulate_ma99_trade(
    df: pd.DataFrame,
    sig: MA200CoilSignal,
    max_hold_bars: int = 288,
    arm_r: float = 1.2,
    scratch_bars: int = 0,
    fee_roundtrip_pct: float = 0.12,
) -> Optional[Dict[str, Any]]:
    """模擬沿 MA99 移動停利，達 +1.2R 後上移保本（已扣 0.12% 費滑）。"""
    i = sig.bar_idx
    if i >= len(df) - 2:
        return None

    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    m99 = df["MA99"].to_numpy(float)
    times = df.index

    entry = sig.entry_price
    cur_stop = sig.stop_price
    risk = entry - cur_stop
    if risk <= 0:
        return None

    end_idx = min(len(df) - 1, i + max_hold_bars)
    armed = False
    exit_idx = end_idx
    exit_px = float(c[end_idx])
    reason = "72h到期"

    for j in range(i + 1, end_idx + 1):
        if l[j] <= cur_stop:
            exit_idx = j
            exit_px = float(cur_stop)
            reason = "保本鎖利" if armed else "初始止損"
            break
        if not armed and h[j] >= entry + arm_r * risk:
            armed = True
            cur_stop = max(cur_stop, entry * 1.0025)
        if scratch_bars > 0 and not armed and (j - i) >= scratch_bars and c[j] >= entry * 1.0025:
            exit_idx = j
            exit_px = float(c[j])
            reason = "盤整微利釋放"
            break
        if armed and np.isfinite(m99[j]) and c[j] < m99[j]:
            exit_idx = j
            exit_px = float(c[j])
            reason = "破MA99止盈"
            break

    pnl_pct = (exit_px / entry - 1.0) * 100.0 - fee_roundtrip_pct
    r_mult = pnl_pct / sig.risk_pct
    return {
        "exit_idx": exit_idx,
        "exit_time": times[exit_idx],
        "exit_price": exit_px,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "r_mult": r_mult,
    }


def plot_signal_chart(
    df: pd.DataFrame,
    signals: Sequence[MA200CoilSignal],
    symbol: str,
    interval: str,
    out_path: Path,
    lookback_bars: int = 160,
    exit_map: Optional[Dict[datetime, Dict[str, Any]]] = None,
) -> None:
    """輸出對齊手機幣安介面 (K線 + 8均線 + 成交量 + RSI) 的復盤圖卡。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if signals:
        max_bar = max(s.bar_idx for s in signals) + 45
        if exit_map:
            ex_indices = [ex["exit_idx"] + 15 for ex in exit_map.values() if "exit_idx" in ex]
            if ex_indices:
                max_bar = max(max_bar, max(ex_indices))
        focus_end = min(len(df), max_bar)
        focus_start = max(0, focus_end - lookback_bars)
        plot_df = df.iloc[focus_start:focus_end].copy()
    else:
        plot_df = df.tail(lookback_bars).copy()

    if plot_df.empty:
        return

    fig, (ax_p, ax_v, ax_r) = plt.subplots(
        3, 1, figsize=(15, 9.4), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0, 0.9], "hspace": 0.06},
        facecolor="#ffffff",
    )

    x = np.arange(len(plot_df))
    o = plot_df["Open"].to_numpy(float)
    h = plot_df["High"].to_numpy(float)
    l = plot_df["Low"].to_numpy(float)
    c = plot_df["Close"].to_numpy(float)
    v = plot_df["Volume"].to_numpy(float)
    times = plot_df.index

    for idx in range(len(plot_df)):
        col = "#0ecb81" if c[idx] >= o[idx] else "#f6465d"
        ax_p.plot([idx, idx], [l[idx], h[idx]], color=col, linewidth=1.0, zorder=2)
        bh = max(abs(c[idx] - o[idx]), (h[idx] - l[idx]) * 0.02, 1e-8)
        ax_p.add_patch(Rectangle((idx - 0.32, min(o[idx], c[idx])), 0.64, bh, facecolor=col, edgecolor=col, zorder=3))
        ax_v.bar(idx, v[idx], width=0.68, color=col, alpha=0.85, zorder=2)

    for period, color, width, label in MA_SPECS:
        col = f"MA{period}"
        if col in plot_df.columns and plot_df[col].notna().any():
            ax_p.plot(x, plot_df[col].to_numpy(float), color=color, linewidth=width, label=label, zorder=4)

    if "VOL_MA5" in plot_df.columns:
        ax_v.plot(x, plot_df["VOL_MA5"].to_numpy(float), color="#f0b90b", linewidth=1.1, label="MA(5)")
    if "VOL_MA10" in plot_df.columns:
        ax_v.plot(x, plot_df["VOL_MA10"].to_numpy(float), color="#7e57c2", linewidth=1.1, label="MA(10)")

    for p, col in ((6, "#f0b90b"), (12, "#ec407a"), (24, "#7e57c2")):
        if f"RSI{p}" in plot_df.columns:
            ax_r.plot(x, plot_df[f"RSI{p}"].to_numpy(float), color=col, linewidth=1.1, label=f"RSI({p})")
    ax_r.axhline(90.0, color="#cbd5e1", linestyle="--", linewidth=0.8)
    ax_r.axhline(50.0, color="#cbd5e1", linestyle="--", linewidth=0.8)
    ax_r.set_ylim(10, 100)

    ts_to_pos = {ts: idx for idx, ts in enumerate(times)}
    y_span = max(float(np.nanmax(h) - np.nanmin(l)), 1e-8)

    for k, sig in enumerate(signals):
        if sig.timestamp not in ts_to_pos:
            continue
        pos = ts_to_pos[sig.timestamp]
        ax_p.scatter([pos], [l[pos] * 0.985], marker="^", s=190, color="#6a1b9a", edgecolors="white", linewidths=1.2, zorder=8)
        extra = f"距MA200: {sig.ext200_pct:+.2f}% | 放量: {sig.qvm:.1f}x | 主買: {sig.taker_buy_pct:.1f}% | 後續最高: +{sig.max_gain_after_pct:.1f}%"
        if exit_map and sig.timestamp in exit_map:
            ex = exit_map[sig.timestamp]
            extra = f"出場: {ex['exit_price']:.6g} ({ex['reason']}) | 1x淨利: {ex['pnl_pct']:+.2f}% ({ex['r_mult']:+.2f}R)"
            if ex["exit_time"] in ts_to_pos:
                epos = ts_to_pos[ex["exit_time"]]
                ax_p.scatter([epos], [h[epos] * 1.015], marker="v", s=150, color="#111827", edgecolors="white", linewidths=1.1, zorder=8)

        msg = (
            f"★ MA200 粘合起漲 (評分 {sig.score})\n"
            f"時間: {sig.timestamp.strftime('%m-%d %H:%M')} | 進場: {sig.entry_price:.6g} | 止損: {sig.stop_price:.6g}\n"
            f"{extra}"
        )
        y_off = l[pos] + (0.24 + 0.14 * (k % 2)) * y_span
        x_text = max(2, pos - 48) if pos > 50 else min(len(plot_df) - 38, pos + 6)
        ax_p.annotate(
            msg,
            xy=(pos, h[pos]),
            xytext=(x_text, min(float(np.nanmax(h)) * 0.94, y_off)),
            fontsize=9.0,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#6a1b9a", "alpha": 0.92, "edgecolor": "none"},
            arrowprops={"arrowstyle": "->", "color": "#6a1b9a", "lw": 1.6},
            zorder=9,
        )

    ax_p.set_title(f"{symbol} ({interval}) MA200 附近均線粘合起漲偵測圖", fontsize=13, fontweight="bold", pad=10)
    ax_p.set_ylabel("價格 (USDT)")
    ax_p.grid(True, linestyle="--", alpha=0.25)
    ax_p.legend(loc="upper left", ncol=4, fontsize=8.3, framealpha=0.85)

    ax_v.set_ylabel("成交量")
    ax_v.grid(True, linestyle="--", alpha=0.25)
    ax_v.legend(loc="upper left", ncol=2, fontsize=8.3)

    ax_r.set_ylabel("RSI")
    ax_r.grid(True, linestyle="--", alpha=0.25)
    ax_r.legend(loc="upper left", ncol=3, fontsize=8.3)

    step = max(1, len(plot_df) // 10)
    ticks = list(range(0, len(plot_df), step))
    ax_r.set_xticks(ticks)
    ax_r.set_xticklabels([times[idx].strftime("%m-%d\n%H:%M") for idx in ticks], fontsize=8.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_signals(signals: Sequence[MA200CoilSignal], exit_map: Optional[Dict[Tuple[str, datetime], Dict[str, Any]]] = None) -> None:
    """格式化輸出偵測到的 MA200 粘合起漲訊號清單。"""
    if not signals:
        print("目前條件與時間窗口內無符合『MA200 附近均線粘合起漲』的訊號。")
        return

    print("=" * 136)
    print(
        f"{'時間 (TPE)':<17} | {'幣種':<12} | {'評分':<4} | {'進場價':<10} | {'止損價':<10} | "
        f"{'距MA200':<8} | {'8線糾結':<8} | {'爆量':<7} | {'主買':<6} | {'後續最高 / 模擬淨利'}"
    )
    print("-" * 136)
    for s in signals:
        tail = f"+{s.max_gain_after_pct:.1f}%"
        if exit_map and (s.symbol, s.timestamp) in exit_map:
            ex = exit_map[(s.symbol, s.timestamp)]
            tail = f"+{s.max_gain_after_pct:.1f}% | {ex['pnl_pct']:+.2f}% ({ex['r_mult']:+.2f}R {ex['reason']})"
        print(
            f"{s.timestamp.strftime('%Y-%m-%d %H:%M'):<17} | {s.symbol:<12} | {s.score:>3}  | "
            f"{s.entry_price:<10.6g} | {s.stop_price:<10.6g} | {s.ext200_pct:>+6.2f}% | "
            f"{s.spread8_pct:>6.2f}% | {s.qvm:>5.1f}x | {s.taker_buy_pct:>4.1f}% | {tail}"
        )
        print(f"   └─ {s.note}")
    print("=" * 136)


def simulate_portfolio(
    signals: Sequence[MA200CoilSignal],
    exit_map: Dict[Tuple[str, datetime], Dict[str, Any]],
    initial_cap: float = 1000.0,
    leverage: float = 3.0,
    risk_pct_per_trade: float = 2.0,
    max_slots: int = 1,
) -> Dict[str, Any]:
    """多槽位/單倉排他滾倉模擬：3x 分槽全倉複利 + 2% 以損定倉安全複利 + 固定 3x 單利。"""
    max_slots = max(1, int(max_slots))
    cash_comp = initial_cap
    cash_risk2 = initial_cap
    eq_fixed = initial_cap
    peak_comp = initial_cap
    peak_risk2 = initial_cap
    mdd_comp = 0.0
    mdd_risk2 = 0.0

    active: List[Dict[str, Any]] = []
    executed_items: List[Tuple[MA200CoilSignal, Dict[str, Any], float, float]] = []
    skipped_items: List[Tuple[MA200CoilSignal, Dict[str, Any]]] = []
    curve_comp: List[float] = [initial_cap]
    curve_risk2: List[float] = [initial_cap]
    curve_fixed: List[float] = [initial_cap]
    labels: List[str] = ["起點"]

    def _settle(cutoff: Optional[datetime]) -> None:
        nonlocal cash_comp, cash_risk2, eq_fixed, peak_comp, peak_risk2, mdd_comp, mdd_risk2
        active.sort(key=lambda item: item["exit_time"])
        while active and (cutoff is None or active[0]["exit_time"] <= cutoff):
            item = active.pop(0)
            cash_comp += item["m_comp"] * (1.0 + item["lev_ret"])
            cash_risk2 += item["m_risk2"] * (1.0 + item["r2_ret"])
            eq_fixed += item["fixed_delta"]

            tot_comp = cash_comp + sum(p["m_comp"] for p in active)
            tot_risk2 = cash_risk2 + sum(p["m_risk2"] for p in active)
            peak_comp = max(peak_comp, tot_comp)
            peak_risk2 = max(peak_risk2, tot_risk2)
            mdd_comp = max(mdd_comp, (peak_comp - tot_comp) / peak_comp * 100.0)
            mdd_risk2 = max(mdd_risk2, (peak_risk2 - tot_risk2) / peak_risk2 * 100.0)
            curve_comp.append(tot_comp)
            curve_risk2.append(tot_risk2)
            curve_fixed.append(eq_fixed)
            sig_obj: MA200CoilSignal = item["sig"]
            labels.append(f"{sig_obj.symbol.replace('USDT', '')}\n{sig_obj.timestamp.strftime('%m-%d')}")
            executed_items.append((sig_obj, item["ex"], item["lev_ret"] * 100.0, tot_comp))

    for s in sorted(signals, key=lambda x: (x.timestamp, -x.score, -x.qvm)):
        ex = exit_map.get((s.symbol, s.timestamp))
        if ex is None:
            continue
        _settle(s.timestamp)
        if len(active) >= max_slots:
            skipped_items.append((s, ex))
            continue

        slots_free = max_slots - len(active)
        m_comp = cash_comp / slots_free
        m_risk2 = cash_risk2 / slots_free
        cash_comp -= m_comp
        cash_risk2 -= m_risk2

        pnl = float(ex["pnl_pct"])
        lev_ret = max(-0.95, pnl / 100.0 * leverage)
        risk_lev = min(leverage, risk_pct_per_trade / max(s.risk_pct, 0.8))
        r2_ret = max(-0.95, (pnl / 100.0) * risk_lev)
        fixed_delta = (initial_cap / max_slots) * (pnl / 100.0 * leverage)

        active.append(
            {
                "exit_time": ex["exit_time"],
                "sig": s,
                "ex": ex,
                "m_comp": m_comp,
                "lev_ret": lev_ret,
                "m_risk2": m_risk2,
                "r2_ret": r2_ret,
                "fixed_delta": fixed_delta,
            }
        )

    _settle(None)

    return {
        "initial_cap": initial_cap,
        "leverage": leverage,
        "max_slots": max_slots,
        "executed": len(executed_items),
        "skipped": len(skipped_items),
        "executed_items": executed_items,
        "skipped_items": skipped_items,
        "eq_comp": cash_comp,
        "comp_ret_pct": (cash_comp / initial_cap - 1.0) * 100.0,
        "eq_risk2": cash_risk2,
        "risk2_ret_pct": (cash_risk2 / initial_cap - 1.0) * 100.0,
        "eq_fixed": eq_fixed,
        "fixed_ret_pct": (eq_fixed / initial_cap - 1.0) * 100.0,
        "mdd_comp": mdd_comp,
        "mdd_risk2": mdd_risk2,
        "curve_comp": curve_comp,
        "curve_risk2": curve_risk2,
        "curve_fixed": curve_fixed,
        "labels": labels,
    }


def run_backtest_cache(cache_dir: Path, preset: str = "sniper", relaxed: bool = False) -> int:
    """針對本地 pickle 快取（如 breakout_output/cache_365d）執行快速無差別回測。"""
    files = sorted(cache_dir.glob("*.pkl"))
    if not files:
        print(f"找不到快取檔案: {cache_dir}")
        return 1

    active_preset = "relaxed" if relaxed else preset
    all_sigs: List[MA200CoilSignal] = []
    exit_map: Dict[Tuple[str, datetime], Dict[str, Any]] = {}

    for path in files:
        sym = path.name.split("_")[0]
        if sym in EXCLUDED_SYMBOLS:
            continue
        raw_df = pd.read_pickle(path)
        if raw_df.empty:
            continue
        df = compute_indicators(raw_df)
        sigs = detect_ma200_coil_breakouts(df, sym, interval="15m", preset=active_preset)
        for s in sigs:
            sim = simulate_ma99_trade(df, s)
            if sim is not None:
                all_sigs.append(s)
                exit_map[(sym, s.timestamp)] = sim

    all_sigs.sort(key=lambda x: x.timestamp)
    print_signals(all_sigs, exit_map=exit_map)
    if not all_sigs:
        return 0

    pnls = np.array([exit_map[(s.symbol, s.timestamp)]["pnl_pct"] for s in all_sigs], dtype=float)
    rs = np.array([exit_map[(s.symbol, s.timestamp)]["r_mult"] for s in all_sigs], dtype=float)
    win_rate = float(np.mean(pnls > 0) * 100.0)
    gp = float(np.sum(pnls[pnls > 0]))
    gl = float(abs(np.sum(pnls[pnls <= 0])))
    pf = gp / gl if gl > 0 else 99.9
    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    port1 = simulate_portfolio(all_sigs, exit_map, max_slots=1)
    port2 = simulate_portfolio(all_sigs, exit_map, max_slots=2)
    print(
        f"\n[回測統計 | 模式={active_preset}] 檔數: {len(files)} | 總訊號: {len(all_sigs)} | 勝率: {win_rate:.1f}% | "
        f"平均淨利: {np.mean(pnls):+.2f}% | 平均R: {np.mean(rs):+.2f}R | 總R: {np.sum(rs):+.1f}R | "
        f"PF: {pf:.2f} | 最大連虧: {max_streak} 筆"
    )
    print(
        f"[單倉模擬 ($1,000 起)] 執行 {port1['executed']} 筆 (略過重疊 {port1['skipped']} 筆) | "
        f"3x全倉複利: ${port1['eq_comp']:,.0f} (MDD {port1['mdd_comp']:.1f}%) | "
        f"2%以損定倉: ${port1['eq_risk2']:,.0f} (MDD {port1['mdd_risk2']:.1f}%) | "
        f"固定3x單利: ${port1['eq_fixed']:,.0f}"
    )
    print(
        f"[雙槽模擬 ($1,000 起)] 執行 {port2['executed']} 筆 (略過重疊 {port2['skipped']} 筆) | "
        f"3x分槽複利: ${port2['eq_comp']:,.0f} (MDD {port2['mdd_comp']:.1f}%) | "
        f"2%以損定倉: ${port2['eq_risk2']:,.0f} (MDD {port2['mdd_risk2']:.1f}%) | "
        f"固定3x單利: ${port2['eq_fixed']:,.0f}"
    )
    return 0


def run_live_scan(
    symbols: Sequence[str],
    interval: str,
    days: int,
    recent_bars: int,
    end_dt: Optional[datetime],
    preset: str,
    relaxed: bool,
    plot: bool,
    out_dir: Path,
    workers: int = 10,
    include_forming: bool = False,
) -> List[MA200CoilSignal]:
    """平行掃描指定幣種清單並選擇性輸出復盤圖卡（實盤預設剔除未收盤 K 線避免訊號重繪）。"""
    found: List[MA200CoilSignal] = []
    drop_forming = (end_dt is None) and not include_forming

    def _worker(sym: str) -> List[MA200CoilSignal]:
        try:
            df = fetch_klines(sym, interval=interval, days=days, end_dt=end_dt, drop_forming=drop_forming)
            sigs = detect_ma200_coil_breakouts(df, sym, interval=interval, preset=preset, relaxed=relaxed)
            if recent_bars > 0:
                min_idx = max(0, len(df) - recent_bars)
                sigs = [s for s in sigs if s.bar_idx >= min_idx]
            if sigs and plot:
                ex_map: Dict[datetime, Dict[str, Any]] = {}
                for s in sigs:
                    sim = simulate_ma99_trade(df, s)
                    if sim is not None:
                        ex_map[s.timestamp] = sim
                plot_signal_chart(df, sigs, sym, interval, out_dir / f"{sym}_{interval}_ma200_coil.png", exit_map=ex_map)
            return sigs
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_worker, symbols):
            if res:
                found.extend(res)

    found.sort(key=lambda s: (s.timestamp, s.score), reverse=True)
    return found


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="幣安永續合約『MA200 附近均線粘合起漲』偵測腳本")
    parser.add_argument("--symbols", type=str, default="", help="指定幣種，逗號分隔 (例如 FILUSDT,DOTUSDT)")
    parser.add_argument("--interval", type=str, default="15m", help="K 線週期 (5m, 15m, 30m, 1h)，預設 15m")
    parser.add_argument("--days", type=int, default=5, help="回溯天數，預設 5 天")
    parser.add_argument("--end-date", type=str, default="", help="歷史復盤截止日期 (YYYY-MM-DD，TPE 時區)")
    parser.add_argument("--scan", action="store_true", help="掃描全市場熱門 USDT 永續合約")
    parser.add_argument("--watch", type=int, default=0, help="持續盯盤輪詢秒數 (例如 --watch 60)")
    parser.add_argument("--top", type=int, default=200, help="全市場掃描前 N 大成交額幣種，預設 200")
    parser.add_argument("--recent-bars", type=int, default=12, help="只顯示最近 N 根 K 線內觸發的新訊號 (0=全部)")
    parser.add_argument("--preset", type=str, default="sniper", choices=["sniper", "balanced", "relaxed"], help="參數模式: sniper(預設嚴選), balanced(平衡擴展), relaxed(寬鬆觀察)")
    parser.add_argument("--relaxed", action="store_true", help="等同 --preset relaxed")
    parser.add_argument("--plot", action="store_true", help="輸出 K 線與 8 均線復盤圖卡 PNG")
    parser.add_argument("--include-forming", action="store_true", help="即時掃描時納入尚未收盤的當根 K 線 (預設僅掃描已收盤 K 線)")
    parser.add_argument("--out-dir", type=str, default="breakout_output", help="圖卡輸出資料夾")
    parser.add_argument("--backtest-cache", type=str, default="", help="指定本地快取資料夾執行回測驗證")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    active_preset = "relaxed" if args.relaxed else args.preset

    if args.backtest_cache:
        return run_backtest_cache(Path(args.backtest_cache), preset=active_preset)

    end_dt: Optional[datetime] = None
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=TPE) + timedelta(days=1)

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        sigs = run_live_scan(
            syms, args.interval, args.days, recent_bars=0, end_dt=end_dt,
            preset=active_preset, relaxed=False, plot=args.plot, out_dir=out_dir,
            workers=min(8, len(syms)), include_forming=args.include_forming,
        )
        print_signals(sigs)
        if args.plot and sigs:
            print(f"[圖表已輸出] {out_dir.resolve()}")
        return 0

    if not args.scan and args.watch <= 0:
        args.scan = True

    seen: Set[Tuple[str, str]] = set()
    while True:
        now_str = datetime.now(TPE).strftime("%Y-%m-%d %H:%M:%S")
        syms = list_usdt_perps(top_n=args.top)
        print(f"\n[{now_str} TPE] 掃描 {len(syms)} 檔永續合約 ({args.interval} | MA200 粘合起漲 {active_preset})...")
        sigs = run_live_scan(
            syms, args.interval, args.days, recent_bars=args.recent_bars, end_dt=end_dt,
            preset=active_preset, relaxed=False, plot=args.plot, out_dir=out_dir,
            include_forming=args.include_forming,
        )
        if args.watch > 0:
            fresh = [s for s in sigs if (s.symbol, s.timestamp.isoformat()) not in seen]
            for s in fresh:
                seen.add((s.symbol, s.timestamp.isoformat()))
            print_signals(fresh if fresh else sigs)
            if fresh:
                print(f"\a[MA200起漲提示] 偵測到 {len(fresh)} 筆新起漲訊號！")
        else:
            print_signals(sigs)
        if args.plot and sigs:
            print(f"[圖表已輸出] {out_dir.resolve()}")
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
