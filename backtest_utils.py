#!/usr/bin/env python3
"""Shared utilities for honest Binance USDT-M futures backtesting.

Provides:
1. Point-in-time (PIT) trailing 24h quote-volume universe ranking & cache loading
   without alphabetical slicing, forced symbols, or today's 24h ticker lookahead.
2. Historical funding-rate fetcher (/fapi/v1/fundingRate) with disk cache and
   configurable conservative 8h fallback estimate when offline/unavailable.
3. Event-driven multi-slot portfolio accounting:
   - Headline: Fixed-fractional risk compounding (e.g. 2% risk/trade) respecting concurrency.
   - 1x real compounded equity curve (plus explicitly labeled arithmetic sum).
   - 3x leveraged slot compounding with maintenance-margin liquidation & market liquidity caps
     (demoted from headline and labeled as theoretical stress test).
4. Frozen-parameter In-Sample vs Out-of-Sample (OOS) split and chronological walk-forward folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
import logging
from pathlib import Path
import pickle
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TPE = ZoneInfo("Asia/Taipei")
BINANCE_FAPI = "https://fapi.binance.com"
FUNDING_INTERVAL_MS = 8 * 3600 * 1000  # 00:00, 08:00, 16:00 UTC

logger = logging.getLogger("backtest_utils")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.WARNING)

_API_LOCK = threading.Lock()
_COOLDOWN_UNTIL = 0.0


def fapi_get_json(
    session: requests.Session,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    base_url: str = BINANCE_FAPI,
    retries: int = 4,
    timeout: int = 20,
) -> Any:
    """Rate-limit-aware GET helper for https://fapi.binance.com with logged retries."""
    global _COOLDOWN_UNTIL
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        wait_cd = 0.0
        with _API_LOCK:
            now_m = time.monotonic()
            if _COOLDOWN_UNTIL > now_m:
                wait_cd = _COOLDOWN_UNTIL - now_m
        if wait_cd > 0:
            time.sleep(wait_cd)
        try:
            resp = session.get(base_url + path, params=params, timeout=timeout)
            w_hdr = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if w_hdr and w_hdr.isdigit() and int(w_hdr) >= 1950:
                with _API_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + 10.0)
            if resp.status_code in (418, 429):
                retry_hdr = resp.headers.get("Retry-After")
                wait_s = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else 2.0 * (2 ** attempt)
                with _API_LOCK:
                    _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, time.monotonic() + wait_s)
                logger.warning(
                    "Binance API 限流 (HTTP %s, %s)，等待 %.1fs 後重試 (%d/%d)",
                    resp.status_code,
                    path,
                    wait_s,
                    attempt + 1,
                    retries,
                )
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            wait_s = 0.5 * (2 ** attempt)
            if attempt + 1 < retries:
                logger.warning("Binance API 請求失敗 (%s): %s，%.1fs 後重試 (%d/%d)", path, exc, wait_s, attempt + 1, retries)
                time.sleep(wait_s)
    raise RuntimeError(f"Binance API failed ({path}): {last_err}")


def list_historical_usdt_perps(
    session: requests.Session,
    start_ms: int,
    end_ms: int,
    excluded_symbols: Iterable[str] = (),
    base_url: str = BINANCE_FAPI,
) -> List[str]:
    """List USDT-M perpetual contracts that existed during [start_ms, end_ms].

    Does NOT require today's status == 'TRADING' and does NOT filter by today's 24h
    volume ticker, avoiding survivorship and future-liquidity lookahead bias.
    """
    excluded = set(excluded_symbols)
    info = fapi_get_json(session, "/fapi/v1/exchangeInfo", base_url=base_url)
    syms: List[str] = []
    for item in info.get("symbols", []):
        sym = item.get("symbol", "")
        if not sym or sym in excluded:
            continue
        if item.get("quoteAsset") != "USDT" or item.get("contractType") != "PERPETUAL":
            continue
        onboard_ms = int(item.get("onboardDate") or 0)
        delivery_ms = int(item.get("deliveryDate") or 0)
        if onboard_ms > 0 and onboard_ms > end_ms:
            continue
        if delivery_ms > 0 and delivery_ms < start_ms:
            continue
        syms.append(sym)
    return sorted(set(syms))


def load_cached_frames(
    cache_dirs: Sequence[Optional[Path]],
    interval: str = "15m",
    excluded_symbols: Iterable[str] = (),
    npz_converter: Optional[Callable[[Path], pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all cached symbol DataFrames across one or more directories without alphabetical truncation."""
    excluded = set(excluded_symbols)
    loaded: Dict[str, pd.DataFrame] = {}

    for cdir in cache_dirs:
        if cdir is None or str(cdir).strip() == "" or not cdir.exists():
            continue
        if cdir.is_file():
            files = [cdir]
        else:
            files = sorted(cdir.glob("*.pkl")) + sorted(cdir.glob("*.npz"))

        for fpath in files:
            if fpath.name.endswith("_funding.pkl"):
                continue
            if fpath.suffix == ".pkl" and fpath.name.startswith("dfs_"):
                try:
                    blob = pickle.loads(fpath.read_bytes())
                    if isinstance(blob, dict):
                        for sym, df in blob.items():
                            sym_u = str(sym).upper()
                            if sym_u not in excluded and isinstance(df, pd.DataFrame) and not df.empty:
                                if sym_u not in loaded or len(df) > len(loaded[sym_u]):
                                    loaded[sym_u] = df
                except Exception as exc:
                    logger.warning("讀取合併快取失敗 (%s): %s", fpath, exc)
                continue

            if fpath.suffix == ".pkl":
                parts = fpath.stem.split("_")
                if len(parts) >= 2 and parts[1] != interval and parts[1] in ("1m", "3m", "5m", "15m", "30m", "1h", "4h"):
                    continue
                sym = parts[0].upper()
                if sym in excluded or not sym.endswith("USDT"):
                    continue
                try:
                    df = pd.read_pickle(fpath)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if sym not in loaded or len(df) > len(loaded[sym]):
                            loaded[sym] = df
                except Exception as exc:
                    logger.warning("讀取快取失敗 (%s): %s", fpath, exc)

            elif fpath.suffix == ".npz" and npz_converter is not None:
                sym = fpath.stem.split("_")[0].upper()
                if sym in excluded or not sym.endswith("USDT"):
                    continue
                try:
                    df = npz_converter(fpath)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        if sym not in loaded or len(df) > len(loaded[sym]):
                            loaded[sym] = df
                except Exception as exc:
                    logger.warning("讀取 NPZ 快取失敗 (%s): %s", fpath, exc)

    return loaded


def extract_quote_volume_array(df: pd.DataFrame) -> np.ndarray:
    """Return per-bar USD quote volume array, falling back to Close * Volume if needed."""
    for col in ("QuoteVol", "QuoteVolume", "quote_vol", "quoteVolume"):
        if col in df.columns:
            arr = df[col].to_numpy(dtype=float)
            return np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    if "Close" in df.columns and "Volume" in df.columns:
        arr = df["Close"].to_numpy(dtype=float) * df["Volume"].to_numpy(dtype=float)
        return np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    return np.zeros(len(df), dtype=float)


def ts_to_ms(ts: datetime | pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).timestamp() * 1000)


class PointInTimeVolumeIndex:
    """Zero-lookahead point-in-time trailing USD quote-volume index across a symbol pool."""

    def __init__(self, dfs: Dict[str, pd.DataFrame]) -> None:
        self._series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._rank_cache: Dict[Tuple[int, int], Dict[str, Tuple[float, int]]] = {}
        for sym, df in dfs.items():
            if df is None or df.empty:
                continue
            sdf = df[~df.index.duplicated(keep="last")].sort_index()
            idx_utc = pd.DatetimeIndex(sdf.index).tz_convert("UTC")
            ts_ms = (idx_utc.asi8 // 1_000_000).to_numpy(dtype=np.int64)
            qv = extract_quote_volume_array(sdf)
            cum_qv = np.empty(len(qv) + 1, dtype=np.float64)
            cum_qv[0] = 0.0
            np.cumsum(qv, out=cum_qv[1:])
            self._series[sym] = (ts_ms, cum_qv)

    @property
    def symbols(self) -> List[str]:
        return sorted(self._series.keys())

    def trailing_qv(self, symbol: str, at_open_ms: int, window_ms: int = 86_400_000) -> float:
        """Trailing quote volume strictly prior to at_open_ms over [at_open_ms - window_ms, at_open_ms)."""
        pair = self._series.get(symbol)
        if pair is None:
            return 0.0
        ts_ms, cum_qv = pair
        r_idx = int(np.searchsorted(ts_ms, at_open_ms, side="left"))
        l_idx = int(np.searchsorted(ts_ms, at_open_ms - window_ms, side="left"))
        if r_idx <= l_idx:
            return 0.0
        return float(max(0.0, cum_qv[r_idx] - cum_qv[l_idx]))

    def cross_section_at(self, at_open_ms: int, window_ms: int = 86_400_000) -> Dict[str, Tuple[float, int]]:
        """Return {symbol: (trailing_qv_usd, rank_1_based)} among symbols active at at_open_ms."""
        cache_key = (at_open_ms, window_ms)
        cached = self._rank_cache.get(cache_key)
        if cached is not None:
            return cached

        active: List[Tuple[float, str]] = []
        for sym in self._series:
            qv = self.trailing_qv(sym, at_open_ms, window_ms=window_ms)
            if qv > 0.0:
                active.append((qv, sym))
        active.sort(key=lambda pair: (-pair[0], pair[1]))
        result = {sym: (qv, rank) for rank, (qv, sym) in enumerate(active, start=1)}
        if len(self._rank_cache) > 4096:
            self._rank_cache.clear()
        self._rank_cache[cache_key] = result
        return result

    def check_eligibility(
        self,
        symbol: str,
        at_ts: datetime | pd.Timestamp,
        min_quote_vol: float = 0.0,
        top_n: int = 0,
        window_ms: int = 86_400_000,
    ) -> Tuple[bool, float, int]:
        """Verify whether symbol meets point-in-time min_quote_vol and top_n rank right before at_ts."""
        at_ms = ts_to_ms(at_ts)
        if top_n <= 0:
            qv = self.trailing_qv(symbol, at_ms, window_ms=window_ms)
            ok = qv >= min_quote_vol and (qv > 0.0 or min_quote_vol <= 0.0)
            return ok, qv, 0

        cross = self.cross_section_at(at_ms, window_ms=window_ms)
        info = cross.get(symbol)
        if info is None:
            return False, 0.0, 0
        qv, rank = info
        ok = (qv >= min_quote_vol) and (rank <= top_n)
        return ok, qv, rank


class FundingRateStore:
    """Historical funding-rate fetcher (/fapi/v1/fundingRate) with local cache & conservative fallback."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        fallback_rate_8h_pct: float = 0.03,
        fetch_online: bool = True,
        session: Optional[requests.Session] = None,
        base_url: str = BINANCE_FAPI,
    ) -> None:
        self.cache_dir = (cache_dir / "funding") if cache_dir is not None else None
        self.fallback_rate_8h_pct = float(fallback_rate_8h_pct)
        self.fetch_online = bool(fetch_online)
        self.session = session or requests.Session()
        self.base_url = base_url
        self._mem: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._fetched_ranges: Dict[str, Tuple[int, int]] = {}
        self._online_disabled = False
        self._lock = threading.Lock()

    def _cache_path(self, symbol: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{symbol.upper()}_funding.pkl"

    def _load_symbol_disk(self, symbol: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        cpath = self._cache_path(symbol)
        if cpath is None or not cpath.is_file():
            return None
        try:
            df = pd.read_pickle(cpath)
            if isinstance(df, pd.DataFrame) and not df.empty and {"fundingTime", "fundingRate"}.issubset(df.columns):
                sdf = df.drop_duplicates(subset=["fundingTime"], keep="last").sort_values("fundingTime")
                return (
                    sdf["fundingTime"].to_numpy(dtype=np.int64),
                    sdf["fundingRate"].to_numpy(dtype=np.float64),
                )
        except Exception as exc:
            logger.warning("讀取資金費率快取失敗 (%s): %s", cpath, exc)
        return None

    def _save_symbol_disk(self, symbol: str, times_ms: np.ndarray, rates: np.ndarray) -> None:
        cpath = self._cache_path(symbol)
        if cpath is None or len(times_ms) == 0:
            return
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({"fundingTime": times_ms, "fundingRate": rates})
            df.to_pickle(cpath)
        except Exception as exc:
            logger.warning("寫入資金費率快取失敗 (%s): %s", cpath, exc)

    def ensure_symbol_range(self, symbol: str, start_ms: int, end_ms: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        sym = symbol.upper()
        with self._lock:
            if sym not in self._mem:
                disk = self._load_symbol_disk(sym)
                if disk is not None:
                    self._mem[sym] = disk

            existing = self._mem.get(sym)
            if existing is not None and len(existing[0]) > 0:
                if existing[0][0] <= start_ms + FUNDING_INTERVAL_MS and existing[0][-1] >= end_ms - FUNDING_INTERVAL_MS:
                    return existing

            prev_rng = self._fetched_ranges.get(sym)
            if prev_rng is not None and prev_rng[0] <= start_ms and prev_rng[1] >= end_ms:
                return existing

            if not self.fetch_online or self._online_disabled:
                return existing

        # Paginate /fapi/v1/fundingRate outside lock
        rows: List[Tuple[int, float]] = []
        cursor_ms = max(0, start_ms - FUNDING_INTERVAL_MS)
        target_end_ms = end_ms + FUNDING_INTERVAL_MS
        try:
            while cursor_ms <= target_end_ms:
                chunk = fapi_get_json(
                    self.session,
                    "/fapi/v1/fundingRate",
                    params={"symbol": sym, "startTime": cursor_ms, "endTime": target_end_ms, "limit": 1000},
                    base_url=self.base_url,
                    retries=2,
                    timeout=12,
                )
                if not isinstance(chunk, list) or not chunk:
                    break
                for item in chunk:
                    ft = int(item.get("fundingTime") or 0)
                    fr = float(item.get("fundingRate") or 0.0)
                    if ft > 0:
                        rows.append((ft, fr))
                last_ft = int(chunk[-1].get("fundingTime") or 0)
                if len(chunk) < 1000 or last_ft <= cursor_ms:
                    break
                cursor_ms = last_ft + 1
                time.sleep(0.03)
        except Exception as exc:
            with self._lock:
                if not self._online_disabled:
                    logger.warning(
                        "無法連線取得歷史資金費率 (%s: %s)，自動切換為保守估計費率 (%+.3f%% / 8h)",
                        sym,
                        exc,
                        self.fallback_rate_8h_pct,
                    )
                    self._online_disabled = True
            return self._mem.get(sym)

        with self._lock:
            self._fetched_ranges[sym] = (start_ms, end_ms)
            if rows:
                merged: Dict[int, float] = {}
                old = self._mem.get(sym)
                if old is not None:
                    for ft, fr in zip(old[0].tolist(), old[1].tolist()):
                        merged[int(ft)] = float(fr)
                for ft, fr in rows:
                    merged[int(ft)] = float(fr)
                sorted_fts = np.array(sorted(merged.keys()), dtype=np.int64)
                sorted_frs = np.array([merged[int(k)] for k in sorted_fts], dtype=np.float64)
                self._mem[sym] = (sorted_fts, sorted_frs)
                self._save_symbol_disk(sym, sorted_fts, sorted_frs)
            return self._mem.get(sym)

    @staticmethod
    def count_8h_settlements(actual_entry_ms: int, actual_exit_ms: int) -> int:
        if actual_exit_ms <= actual_entry_ms:
            return 0
        return int((actual_exit_ms // FUNDING_INTERVAL_MS) - (actual_entry_ms // FUNDING_INTERVAL_MS))

    def compute_funding_cost_pct(
        self,
        symbol: str,
        entry_bar_open_ms: int,
        exit_bar_open_ms: int,
        bar_ms: int = 900_000,
    ) -> Tuple[float, int, str]:
        """Compute long funding cost (% of notional) across (entry_close_ms, exit_close_ms].

        Positive return value = cost deducted from long PnL; negative = net funding rebate received.
        """
        actual_entry_ms = int(entry_bar_open_ms) + int(bar_ms)
        actual_exit_ms = int(exit_bar_open_ms) + int(bar_ms)
        if actual_exit_ms <= actual_entry_ms:
            return 0.0, 0, "none"

        data = self.ensure_symbol_range(symbol, actual_entry_ms, actual_exit_ms)
        if data is not None and len(data[0]) > 0:
            fts, frs = data
            # Verify data actually spans near this window rather than a disjoint era
            if fts[0] <= actual_entry_ms + FUNDING_INTERVAL_MS and fts[-1] >= actual_exit_ms - FUNDING_INTERVAL_MS:
                l_idx = int(np.searchsorted(fts, actual_entry_ms, side="right"))
                r_idx = int(np.searchsorted(fts, actual_exit_ms, side="right"))
                n_events = max(0, r_idx - l_idx)
                cost_pct = float(np.sum(frs[l_idx:r_idx]) * 100.0) if n_events > 0 else 0.0
                return cost_pct, n_events, "fapi"

        n_8h = self.count_8h_settlements(actual_entry_ms, actual_exit_ms)
        return float(n_8h * self.fallback_rate_8h_pct), n_8h, "est"


def approx_maintenance_margin_rate(notional_usd: float, base_mmr: float = 0.015) -> float:
    """Approximate tiered Binance USDT-M altcoin maintenance margin rate + clearance buffer."""
    if notional_usd >= 1_000_000.0:
        return max(base_mmr, 0.10)
    if notional_usd >= 250_000.0:
        return max(base_mmr, 0.05)
    if notional_usd >= 50_000.0:
        return max(base_mmr, 0.025)
    return max(0.005, float(base_mmr))


def check_long_liquidation(
    entry_price: float,
    worst_low_price: float,
    effective_leverage: float,
    notional_usd: float,
    base_mmr: float = 0.015,
    taker_fee_one_way: float = 0.0006,
) -> Tuple[bool, float]:
    """Return (is_liquidated, liq_price) for an isolated long position."""
    if effective_leverage <= 1.0001 or entry_price <= 0.0 or notional_usd <= 0.0:
        return False, 0.0
    mmr = approx_maintenance_margin_rate(notional_usd, base_mmr=base_mmr)
    max_loss_frac = (1.0 / effective_leverage) - mmr - taker_fee_one_way
    if max_loss_frac <= 0.0:
        return True, entry_price
    liq_price = entry_price * (1.0 - max_loss_frac)
    return bool(worst_low_price <= liq_price), float(liq_price)


@dataclass
class PortfolioTradeInput:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    stop_price: float
    exit_price: float
    risk_pct: float
    net_pnl_pct: float
    r_mult: float
    worst_low_price: float
    bar_quote_vol_usd: float = 0.0
    trailing_24h_qv_usd: float = 0.0
    funding_cost_pct: float = 0.0
    priority: float = 0.0
    payload: Any = None


@dataclass
class ExecutedPositionRecord:
    trade: PortfolioTradeInput
    notional_usd: float
    margin_usd: float
    pnl_usd: float
    acct_ret_pct: float
    equity_after_usd: float
    liquidity_capped: bool
    liquidated: bool


def _compute_liquidity_cap_usd(
    t: PortfolioTradeInput,
    max_bar_vol_share: float,
    max_24h_vol_share: float,
    hard_notional_cap_usd: float,
) -> float:
    caps: List[float] = []
    if hard_notional_cap_usd > 0:
        caps.append(hard_notional_cap_usd)
    if max_bar_vol_share > 0 and t.bar_quote_vol_usd > 0:
        caps.append(t.bar_quote_vol_usd * max_bar_vol_share)
    if max_24h_vol_share > 0 and t.trailing_24h_qv_usd > 0:
        caps.append(t.trailing_24h_qv_usd * max_24h_vol_share)
    return min(caps) if caps else float("inf")


def _simulate_track(
    ordered_trades: Sequence[PortfolioTradeInput],
    initial_cap: float,
    mode: str,
    *,
    max_positions: int = 3,
    leverage: float = 3.0,
    risk_pct_per_trade: float = 2.0,
    base_mmr: float = 0.015,
    apply_liquidity_cap: bool = True,
    max_bar_vol_share: float = 0.02,
    max_24h_vol_share: float = 0.005,
    hard_notional_cap_usd: float = 500_000.0,
) -> Dict[str, Any]:
    """Event-driven multi-slot account track simulator.

    mode:
      - 'fixed_risk': fixed-fractional risk_pct_per_trade of current equity (headline)
      - 'spot_1x': real 1x unleveraged slot compounding
      - 'comp_lev': leveraged slot compounding with maintenance-margin liquidation & liquidity caps
      - 'fixed_lev': non-compounding fixed-base leveraged sizing
    """
    max_positions = max(1, int(max_positions))
    cash = float(initial_cap)
    peak = float(initial_cap)
    max_dd_pct = 0.0

    active: List[Dict[str, Any]] = []
    executed_records: List[ExecutedPositionRecord] = []
    skipped_concurrency = 0
    liquidity_capped_count = 0
    liquidated_count = 0

    curve_times: List[datetime] = []
    curve_values: List[float] = [float(initial_cap)]
    curve_labels: List[str] = ["起點"]
    if ordered_trades:
        curve_times.append(ordered_trades[0].entry_time)

    def _settle_until(cutoff: Optional[datetime]) -> None:
        nonlocal cash, peak, max_dd_pct
        active.sort(key=lambda item: item["exit_time"])
        while active and (cutoff is None or active[0]["exit_time"] <= cutoff):
            pos = active.pop(0)
            cash = max(0.0, cash + pos["margin_usd"] + pos["pnl_usd"])
            tot_eq = cash + sum(p["margin_usd"] for p in active)
            peak = max(peak, tot_eq)
            if peak > 0:
                dd = (peak - tot_eq) / peak * 100.0
                max_dd_pct = max(max_dd_pct, dd)

            tr_in: PortfolioTradeInput = pos["trade"]
            curve_times.append(pos["exit_time"])
            curve_values.append(tot_eq)
            curve_labels.append(f"{tr_in.symbol.replace('USDT', '')}\n{tr_in.entry_time.strftime('%m-%d')}")
            acct_ret_pct = (pos["pnl_usd"] / max(pos["equity_before"], 1e-9)) * 100.0
            executed_records.append(
                ExecutedPositionRecord(
                    trade=tr_in,
                    notional_usd=pos["notional_usd"],
                    margin_usd=pos["margin_usd"],
                    pnl_usd=pos["pnl_usd"],
                    acct_ret_pct=acct_ret_pct,
                    equity_after_usd=tot_eq,
                    liquidity_capped=pos["liquidity_capped"],
                    liquidated=pos["liquidated"],
                )
            )

    for tr in ordered_trades:
        _settle_until(tr.entry_time)
        if len(active) >= max_positions:
            skipped_concurrency += 1
            continue

        equity_now = cash + sum(p["margin_usd"] for p in active)
        if cash <= 1e-6 or equity_now <= 1e-6:
            skipped_concurrency += 1
            continue

        free_slots = max(1, max_positions - len(active))
        liq_cap = (
            _compute_liquidity_cap_usd(tr, max_bar_vol_share, max_24h_vol_share, hard_notional_cap_usd)
            if apply_liquidity_cap
            else float("inf")
        )

        if mode == "fixed_risk":
            risk_usd = equity_now * (risk_pct_per_trade / 100.0)
            stop_frac = max(tr.risk_pct / 100.0, 0.005)
            desired_notional = risk_usd / stop_frac
            open_notional = sum(p["notional_usd"] for p in active)
            acct_lev_cap = max(0.0, equity_now * leverage - open_notional)
            slot_margin_cap = cash * leverage
            max_allowed = min(acct_lev_cap, slot_margin_cap)
            notional_pre_liq = min(desired_notional, max_allowed)
            notional = min(notional_pre_liq, liq_cap)
            liq_capped = notional < notional_pre_liq - 1e-6
            slot_cash = cash / free_slots
            margin_usd = min(cash, max(notional / max(leverage, 1.0), min(slot_cash, notional)))
            eff_pos_lev = notional / max(margin_usd, 1e-9) if margin_usd > 0 else 1.0

        elif mode == "spot_1x":
            slot_cash = cash / free_slots
            notional = min(slot_cash, liq_cap)
            liq_capped = notional < slot_cash - 1e-6
            margin_usd = notional
            eff_pos_lev = 1.0

        elif mode == "comp_lev":
            slot_margin = cash / free_slots
            desired_notional = slot_margin * leverage
            notional = min(desired_notional, liq_cap)
            liq_capped = notional < desired_notional - 1e-6
            margin_usd = notional / max(leverage, 1.0)
            eff_pos_lev = leverage

        elif mode == "fixed_lev":
            base_slot_margin = min(cash / free_slots, initial_cap / max_positions)
            desired_notional = base_slot_margin * leverage
            notional = min(desired_notional, liq_cap)
            liq_capped = notional < desired_notional - 1e-6
            margin_usd = notional / max(leverage, 1.0)
            eff_pos_lev = leverage

        else:
            raise ValueError(f"Unknown portfolio mode: {mode}")

        if notional <= 0.0 or margin_usd <= 0.0:
            skipped_concurrency += 1
            continue

        is_liq, _ = check_long_liquidation(
            entry_price=tr.entry_price,
            worst_low_price=tr.worst_low_price,
            effective_leverage=eff_pos_lev,
            notional_usd=notional,
            base_mmr=base_mmr,
        )
        if is_liq:
            pnl_usd = -margin_usd
            liquidated_count += 1
        else:
            pnl_usd = max(-margin_usd, notional * (tr.net_pnl_pct / 100.0))

        if liq_capped:
            liquidity_capped_count += 1

        cash -= margin_usd
        active.append(
            {
                "exit_time": tr.exit_time,
                "trade": tr,
                "notional_usd": notional,
                "margin_usd": margin_usd,
                "pnl_usd": pnl_usd,
                "equity_before": equity_now,
                "liquidity_capped": liq_capped,
                "liquidated": is_liq,
            }
        )

    _settle_until(None)
    final_eq = cash
    return {
        "initial_cap": initial_cap,
        "final_eq": final_eq,
        "ret_pct": (final_eq / initial_cap - 1.0) * 100.0 if initial_cap > 0 else 0.0,
        "mdd_pct": max_dd_pct,
        "executed_count": len(executed_records),
        "skipped_count": skipped_concurrency,
        "liquidity_capped_count": liquidity_capped_count,
        "liquidated_count": liquidated_count,
        "records": executed_records,
        "curve_times": curve_times,
        "curve_values": curve_values,
        "curve_labels": curve_labels,
    }


def simulate_honest_portfolio(
    trades: Sequence[PortfolioTradeInput],
    *,
    initial_cap: float = 1000.0,
    risk_pct_per_trade: float = 2.0,
    max_positions: int = 3,
    leverage_3x: float = 3.0,
    slots_3x: int = 1,
    base_mmr: float = 0.015,
    apply_liquidity_cap: bool = True,
    max_bar_vol_share: float = 0.02,
    max_24h_vol_share: float = 0.005,
    hard_notional_cap_usd: float = 500_000.0,
) -> Dict[str, Any]:
    """Run all portfolio tracks with honest concurrency, liquidity caps, and liquidation checks."""
    ordered = sorted(trades, key=lambda t: (t.entry_time, -t.priority, t.symbol))

    headline_risk = _simulate_track(
        ordered,
        initial_cap,
        "fixed_risk",
        max_positions=max_positions,
        leverage=leverage_3x,
        risk_pct_per_trade=risk_pct_per_trade,
        base_mmr=base_mmr,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
        max_24h_vol_share=max_24h_vol_share,
        hard_notional_cap_usd=hard_notional_cap_usd,
    )
    spot_1x = _simulate_track(
        ordered,
        initial_cap,
        "spot_1x",
        max_positions=max_positions,
        leverage=1.0,
        base_mmr=base_mmr,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
        max_24h_vol_share=max_24h_vol_share,
        hard_notional_cap_usd=hard_notional_cap_usd,
    )
    comp_3x = _simulate_track(
        ordered,
        initial_cap,
        "comp_lev",
        max_positions=slots_3x,
        leverage=leverage_3x,
        base_mmr=base_mmr,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
        max_24h_vol_share=max_24h_vol_share,
        hard_notional_cap_usd=hard_notional_cap_usd,
    )
    fixed_3x = _simulate_track(
        ordered,
        initial_cap,
        "fixed_lev",
        max_positions=slots_3x,
        leverage=leverage_3x,
        base_mmr=base_mmr,
        apply_liquidity_cap=apply_liquidity_cap,
        max_bar_vol_share=max_bar_vol_share,
        max_24h_vol_share=max_24h_vol_share,
        hard_notional_cap_usd=hard_notional_cap_usd,
    )

    arith_sum_1x_pct = float(sum(t.net_pnl_pct for t in ordered))
    total_funding_pct = float(sum(t.funding_cost_pct for t in ordered))

    return {
        "initial_cap": initial_cap,
        "risk_pct_per_trade": risk_pct_per_trade,
        "max_positions": max_positions,
        "leverage_3x": leverage_3x,
        "slots_3x": slots_3x,
        "apply_liquidity_cap": apply_liquidity_cap,
        "max_bar_vol_share": max_bar_vol_share,
        "headline_risk": headline_risk,
        "spot_1x": spot_1x,
        "comp_3x": comp_3x,
        "fixed_3x": fixed_3x,
        "arith_sum_1x_pct": arith_sum_1x_pct,
        "total_funding_pct": total_funding_pct,
    }


def summarize_period_trades(
    trades: Sequence[PortfolioTradeInput],
    *,
    initial_cap: float = 1000.0,
    risk_pct_per_trade: float = 2.0,
    max_positions: int = 3,
) -> Dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "arith_sum_pnl_pct": 0.0,
            "avg_r": 0.0,
            "sum_r": 0.0,
            "pf": 0.0,
            "max_loss_streak": 0,
            "risk_eq": initial_cap,
            "risk_ret_pct": 0.0,
            "risk_mdd_pct": 0.0,
            "spot_eq": initial_cap,
            "spot_ret_pct": 0.0,
            "spot_mdd_pct": 0.0,
            "executed_risk": 0,
            "skipped_risk": 0,
        }

    pnls = np.array([t.net_pnl_pct for t in trades], dtype=float)
    rs = np.array([t.r_mult for t in trades], dtype=float)
    wins = int(np.sum(pnls > 0))
    gp = float(np.sum(pnls[pnls > 0]))
    gl = float(abs(np.sum(pnls[pnls <= 0])))
    pf = (gp / gl) if gl > 0 else float("inf")

    streak = 0
    max_streak = 0
    for p in pnls:
        if p <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    port = simulate_honest_portfolio(
        trades,
        initial_cap=initial_cap,
        risk_pct_per_trade=risk_pct_per_trade,
        max_positions=max_positions,
    )
    hr = port["headline_risk"]
    sp = port["spot_1x"]
    return {
        "n": len(trades),
        "wins": wins,
        "win_rate": float(wins / len(trades) * 100.0),
        "avg_pnl_pct": float(np.mean(pnls)),
        "arith_sum_pnl_pct": float(np.sum(pnls)),
        "avg_r": float(np.mean(rs)),
        "sum_r": float(np.sum(rs)),
        "pf": pf,
        "max_loss_streak": max_streak,
        "risk_eq": hr["final_eq"],
        "risk_ret_pct": hr["ret_pct"],
        "risk_mdd_pct": hr["mdd_pct"],
        "spot_eq": sp["final_eq"],
        "spot_ret_pct": sp["ret_pct"],
        "spot_mdd_pct": sp["mdd_pct"],
        "executed_risk": hr["executed_count"],
        "skipped_risk": hr["skipped_count"],
    }


def compute_oos_and_walkforward(
    trades: Sequence[PortfolioTradeInput],
    window_start: datetime,
    window_end: datetime,
    *,
    oos_split_date: str = "",
    oos_ratio: float = 0.30,
    wf_folds: int = 4,
    initial_cap: float = 1000.0,
    risk_pct_per_trade: float = 2.0,
    max_positions: int = 3,
) -> Dict[str, Any]:
    """Evaluate frozen-parameter In-Sample vs Out-of-Sample and chronological Walk-Forward folds."""
    ws = pd.Timestamp(window_start).tz_convert(TPE) if pd.Timestamp(window_start).tzinfo else pd.Timestamp(window_start, tz=TPE)
    we = pd.Timestamp(window_end).tz_convert(TPE) if pd.Timestamp(window_end).tzinfo else pd.Timestamp(window_end, tz=TPE)
    if we <= ws:
        we = ws + pd.Timedelta(days=1)

    if oos_split_date.strip():
        split_ts = pd.Timestamp(oos_split_date.strip(), tz=TPE)
    else:
        clamped_ratio = min(0.80, max(0.10, float(oos_ratio)))
        split_ts = ws + (we - ws) * (1.0 - clamped_ratio)

    is_trades = [t for t in trades if pd.Timestamp(t.entry_time) < split_ts]
    oos_trades = [t for t in trades if pd.Timestamp(t.entry_time) >= split_ts]

    is_stats = summarize_period_trades(
        is_trades, initial_cap=initial_cap, risk_pct_per_trade=risk_pct_per_trade, max_positions=max_positions
    )
    oos_stats = summarize_period_trades(
        oos_trades, initial_cap=initial_cap, risk_pct_per_trade=risk_pct_per_trade, max_positions=max_positions
    )

    folds: List[Dict[str, Any]] = []
    n_folds = max(0, int(wf_folds))
    if n_folds >= 2:
        span = (we - ws) / n_folds
        for idx in range(n_folds):
            f_start = ws + span * idx
            f_end = we if idx == n_folds - 1 else (ws + span * (idx + 1))
            if idx == n_folds - 1:
                f_trades = [t for t in trades if f_start <= pd.Timestamp(t.entry_time) <= f_end]
            else:
                f_trades = [t for t in trades if f_start <= pd.Timestamp(t.entry_time) < f_end]
            st = summarize_period_trades(
                f_trades, initial_cap=initial_cap, risk_pct_per_trade=risk_pct_per_trade, max_positions=max_positions
            )
            folds.append(
                {
                    "fold": idx + 1,
                    "start_str": f_start.strftime("%Y-%m-%d"),
                    "end_str": f_end.strftime("%Y-%m-%d"),
                    "stats": st,
                }
            )

    return {
        "window_start_str": ws.strftime("%Y-%m-%d"),
        "window_end_str": we.strftime("%Y-%m-%d"),
        "split_ts": split_ts,
        "split_str": split_ts.strftime("%Y-%m-%d %H:%M"),
        "in_sample": is_stats,
        "out_of_sample": oos_stats,
        "folds": folds,
    }


def fmt_pf(pf: float) -> str:
    if not np.isfinite(pf):
        return "∞ (無虧損)"
    return f"{pf:.2f}"


def render_oos_walkforward_html(oos_report: Dict[str, Any], risk_pct: float = 2.0) -> str:
    """Render HTML section comparing frozen-parameter In-Sample vs Out-of-Sample & Walk-Forward folds."""
    ins = oos_report["in_sample"]
    oos = oos_report["out_of_sample"]
    ws = escape(oos_report["window_start_str"])
    we = escape(oos_report["window_end_str"])
    sp = escape(oos_report["split_str"][:10])

    def _row(label: str, period_str: str, st: Dict[str, Any], highlight: bool = False) -> str:
        bg = " style='background:rgba(56,189,248,0.08);'" if highlight else ""
        ret_cls = "pos" if st["risk_ret_pct"] >= 0 else "neg"
        r_cls = "pos" if st["sum_r"] >= 0 else "neg"
        return (
            f"<tr{bg}>"
            f"<td><b>{escape(label)}</b></td>"
            f"<td>{escape(period_str)}</td>"
            f"<td>{st['n']} 筆 ({st['wins']}勝)</td>"
            f"<td>{st['win_rate']:.1f}%</td>"
            f"<td class='{r_cls}'><b>{st['avg_r']:+.2f}R</b></td>"
            f"<td class='{r_cls}'><b>{st['sum_r']:+.1f}R</b></td>"
            f"<td>{escape(fmt_pf(st['pf']))}</td>"
            f"<td class='{ret_cls}'><b>${st['risk_eq']:,.0f} ({st['risk_ret_pct']:+.1f}%)</b></td>"
            f"<td>-{st['risk_mdd_pct']:.1f}%</td>"
            f"<td>${st['spot_eq']:,.0f} ({st['spot_ret_pct']:+.1f}%)</td>"
            f"</tr>"
        )

    fold_rows = "".join(
        _row(f"WF 第 {f['fold']} 期", f"{f['start_str']} ~ {f['end_str']}", f["stats"])
        for f in oos_report.get("folds", [])
    )
    wf_table = ""
    if fold_rows:
        wf_table = f"""
        <h3 style="font-size:15px;margin:16px 0 8px;color:#cbd5e1;">滾動時間切片 Walk-Forward 穩定度檢驗（參數全程凍結）</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>切片區間</th><th>日期範圍</th><th>交易數</th><th>勝率</th><th>單筆期望 R</th>
                <th>累積 R</th><th>獲利因子 (PF)</th><th>{risk_pct:g}% 定倉獨立淨值</th><th>最大回撤</th><th>1x 現貨複利</th>
              </tr>
            </thead>
            <tbody>{fold_rows}</tbody>
          </table>
        </div>
        """

    return f"""
    <h2>🧪 樣本內 (In-Sample) vs 樣本外盲測 (Out-of-Sample) 獨立檢驗（參數凍結）</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>檢驗區間</th><th>日期範圍 (TPE)</th><th>交易數</th><th>勝率</th><th>單筆期望 R</th>
            <th>累積 R</th><th>獲利因子 (PF)</th><th>{risk_pct:g}% 以損定倉獨立淨值</th><th>最大回撤</th><th>1x 真實複利淨值</th>
          </tr>
        </thead>
        <tbody>
          {_row("樣本內 (In-Sample)", f"{ws} ~ {sp}", ins, highlight=False)}
          {_row("樣本外盲測 (Out-of-Sample)", f"{sp} ~ {we}", oos, highlight=True)}
        </tbody>
      </table>
    </div>
    {wf_table}
    """
