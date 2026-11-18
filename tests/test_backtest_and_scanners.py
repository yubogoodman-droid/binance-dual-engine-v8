from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest_180d_squeeze as bt_sq
import backtest_365d_ma200 as bt_ma200
import backtest_90d_breakout as bt_90
import backtest_utils as bt_utils
import binance_breakout_scanner as brk_scanner
import ma200_coil_scanner as ma200_scanner
import squeeze_breakout_radar as sq_radar

TPE = ZoneInfo("Asia/Taipei")


def _make_raw_klines(n: int = 275, bar_ms: int = 900_000, include_forming: bool = True) -> List[List[Any]]:
    now_ms = int(time.time() * 1000)
    current_bar_open = (now_ms // bar_ms) * bar_ms
    last_open = current_bar_open if include_forming else (current_bar_open - bar_ms)
    start_open = last_open - (n - 1) * bar_ms

    rows: List[List[Any]] = []
    for idx in range(n):
        t_open = start_open + idx * bar_ms
        # Flat tight range on closed bars; huge spike ONLY on the forming bar (last bar)
        if include_forming and idx == n - 1:
            o, h, l, c = 100.0, 108.0, 99.9, 107.5
            vol, qv, tbv = 50_000.0, 5_200_000.0, 38_000.0
        else:
            drift = 100.0 + 0.004 * idx
            o = drift
            h = drift * 1.002
            l = drift * 0.998
            c = drift * 1.0005
            vol = 1_000.0
            qv = 100_000.0
            tbv = 520.0
        rows.append(
            [
                t_open,
                f"{o:.6f}",
                f"{h:.6f}",
                f"{l:.6f}",
                f"{c:.6f}",
                f"{vol:.4f}",
                t_open + bar_ms - 1,
                f"{qv:.4f}",
                100,
                f"{tbv:.4f}",
                f"{tbv * c:.4f}",
                "0",
            ]
        )
    return rows


def test_official_fapi_endpoint_and_no_hardcoded_biases() -> None:
    assert brk_scanner.BASE == "https://fapi.binance.com"
    assert sq_radar.BINANCE_FAPI == "https://fapi.binance.com"
    assert ma200_scanner.BINANCE_FAPI == "https://fapi.binance.com"

    sq_src = Path(bt_sq.__file__).read_text(encoding="utf-8")
    b90_src = Path(bt_90.__file__).read_text(encoding="utf-8")
    assert "EVAAUSDT" not in sq_src
    assert "AKEUSDT" not in b90_src
    assert "C:\\Users" not in b90_src
    assert "C:\\tmp" not in b90_src
    assert "+0.6R" not in sq_src
    assert "+0.55R" in sq_src or "BREAKEVEN_ARM_R" in sq_src


def test_live_scanners_drop_forming_candle_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_raw_klines(n=275, include_forming=True)
    forming_open_ms = int(raw[-1][0])

    monkeypatch.setattr(sq_radar, "api_get", lambda path, params=None, retries=4: list(raw))
    df_sq = sq_radar.load_klines_with_orderflow("TESTUSDT", interval="15m", limit=270)
    assert not df_sq.empty
    assert bt_utils.ts_to_ms(df_sq.index[-1]) < forming_open_ms

    monkeypatch.setattr(brk_scanner, "get_json", lambda path, params=None, retries=4: list(raw))
    df_brk = brk_scanner.fetch_klines("TESTUSDT", interval="15m", days=2)
    assert not df_brk.empty
    assert bt_utils.ts_to_ms(df_brk.index[-1]) < forming_open_ms

    calls = {"n": 0}

    def _fake_ma200_get(path: str, params: Any = None, retries: int = 4) -> Any:
        calls["n"] += 1
        return list(raw) if calls["n"] == 1 else []

    monkeypatch.setattr(ma200_scanner, "api_get", _fake_ma200_get)
    df_ma = ma200_scanner.fetch_klines("TESTUSDT", interval="15m", days=1)
    assert not df_ma.empty
    assert bt_utils.ts_to_ms(df_ma.index[-1]) < forming_open_ms


def test_watch_seen_persistence_across_restarts(tmp_path: Path) -> None:
    for mod in (sq_radar, brk_scanner, ma200_scanner):
        state_file = tmp_path / f"{mod.__name__}_seen.json"
        seen = {("BTCUSDT", "2026-10-01T12:00:00+08:00"), ("ETHUSDT", "2026-10-01T12:15:00+08:00")}
        mod.save_seen_state(state_file, seen)
        reloaded = mod.load_seen_state(state_file)
        assert reloaded == seen


def test_rate_limit_retry_and_logged_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0", "X-MBX-USED-WEIGHT-1M": "100"}

    resp_ok = MagicMock()
    resp_ok.status_code = 200
    resp_ok.headers = {"X-MBX-USED-WEIGHT-1M": "110"}
    resp_ok.raise_for_status.return_value = None
    resp_ok.json.return_value = {"ok": True}

    seq = [resp_429, resp_ok]
    monkeypatch.setattr(sq_radar.HTTP, "get", lambda *a, **kw: seq.pop(0))
    monkeypatch.setattr(time, "sleep", lambda s: None)

    with caplog.at_level("WARNING"):
        out = sq_radar.api_get("/fapi/v1/ping")
    assert out == {"ok": True}
    assert any("429" in rec.message for rec in caplog.records)


def test_pit_volume_ranking_and_historical_universe() -> None:
    idx = pd.date_range("2026-06-01", periods=200, freq="15min", tz=TPE)
    # EARLYUSDT has high volume in first 100 bars, dies later; LATEUSDT is tiny early, pumps after bar 120
    early_qv = np.where(np.arange(200) < 110, 100_000.0, 1_000.0)
    late_qv = np.where(np.arange(200) < 110, 2_000.0, 500_000.0)

    dfs = {
        "EARLYUSDT": pd.DataFrame({"Close": 1.0, "Volume": early_qv, "QuoteVol": early_qv}, index=idx),
        "LATEUSDT": pd.DataFrame({"Close": 1.0, "Volume": late_qv, "QuoteVol": late_qv}, index=idx),
    }
    pit = bt_utils.PointInTimeVolumeIndex(dfs)

    # At bar 100 (before LATEUSDT pumped), EARLYUSDT must rank #1 and LATEUSDT must NOT hijack #1 via future volume
    ok_early, qv_e, rank_e = pit.check_eligibility("EARLYUSDT", idx[100], min_quote_vol=1_000_000.0, top_n=1)
    ok_late, qv_l, rank_l = pit.check_eligibility("LATEUSDT", idx[100], min_quote_vol=1_000_000.0, top_n=1)
    assert ok_early and rank_e == 1 and qv_e > 5_000_000.0
    assert not ok_late and rank_l == 2 and qv_l < 500_000.0

    # Mock exchangeInfo with a settled/delisted perpetual that WAS active during the window
    fake_session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.raise_for_status.return_value = None
    w_start = bt_utils.ts_to_ms(idx[0])
    w_end = bt_utils.ts_to_ms(idx[-1])
    resp.json.return_value = {
        "symbols": [
            {
                "symbol": "DELISTEDUSDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "SETTLING",
                "onboardDate": w_start - 86_400_000 * 30,
                "deliveryDate": w_start + 86_400_000,
            },
            {
                "symbol": "FUTURECOINUSDT",
                "quoteAsset": "USDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "onboardDate": w_end + 86_400_000 * 10,
                "deliveryDate": 4133404800000,
            },
        ]
    }
    fake_session.get.return_value = resp
    hist_syms = bt_utils.list_historical_usdt_perps(fake_session, w_start, w_end)
    assert "DELISTEDUSDT" in hist_syms
    assert "FUTURECOINUSDT" not in hist_syms


def test_funding_rate_cost_online_cache_and_fallback(tmp_path: Path) -> None:
    # 07:45 UTC bar closes at 08:00 UTC; holding until 16:00 UTC bar (closes 16:15 UTC) crosses 16:00 UTC settlement
    t0 = pd.Timestamp("2026-07-01 07:45:00", tz="UTC")
    t1 = pd.Timestamp("2026-07-01 16:00:00", tz="UTC")
    # Entry happens at close of t0 = 08:00 UTC; exit at close of t1 = 16:15 UTC -> crosses 1 settlement (16:00 UTC)
    store_offline = bt_utils.FundingRateStore(cache_dir=tmp_path, fallback_rate_8h_pct=0.04, fetch_online=False)
    cost_est, n_ev, src = store_offline.compute_funding_cost_pct(
        "BTCUSDT", bt_utils.ts_to_ms(t0), bt_utils.ts_to_ms(t1), bar_ms=900_000
    )
    assert src == "est"
    assert n_ev == 1
    assert pytest.approx(cost_est, rel=1e-6) == 0.04

    # Cached historical funding file takes precedence
    fdir = tmp_path / "funding"
    fdir.mkdir(parents=True, exist_ok=True)
    ts_1600 = int(pd.Timestamp("2026-07-01 16:00:00", tz="UTC").timestamp() * 1000)
    pd.DataFrame(
        {
            "fundingTime": [ts_1600 - 28_800_000, ts_1600, ts_1600 + 28_800_000],
            "fundingRate": [0.0001, 0.00075, 0.0002],
        }
    ).to_pickle(fdir / "ETHUSDT_funding.pkl")
    store_cached = bt_utils.FundingRateStore(cache_dir=tmp_path, fallback_rate_8h_pct=0.04, fetch_online=False)
    cost_fapi, n_fapi, src_fapi = store_cached.compute_funding_cost_pct(
        "ETHUSDT", bt_utils.ts_to_ms(t0), bt_utils.ts_to_ms(t1), bar_ms=900_000
    )
    assert src_fapi == "fapi"
    assert n_fapi == 1
    assert pytest.approx(cost_fapi, rel=1e-6) == 0.075


def test_conservative_intrabar_breakeven_stop_across_all_simulators() -> None:
    idx = pd.date_range("2026-08-01", periods=20, freq="15min", tz=TPE)
    opens = np.full(20, 100.0)
    highs = np.full(20, 101.0)
    lows = np.full(20, 99.0)
    closes = np.full(20, 100.5)

    # Signal at bar 5: entry = 100, initial stop = 95 (risk = 5).
    # On bar 6: High spikes to 107 (+1.4R, arming breakeven at ~100.25-100.30),
    # AND Low drops to 99.5 (above initial stop 95, but BELOW newly raised breakeven stop 100.25/100.30!),
    # while later bars rally to 130 so non-conservative code would book a huge fake win!
    highs[6] = 107.0
    lows[6] = 99.5
    closes[6] = 102.0
    for k in range(7, 20):
        opens[k] = 120.0
        highs[k] = 132.0
        lows[k] = 119.0
        closes[k] = 130.0

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.full(20, 1000.0),
            "QuoteVol": np.full(20, 100_000.0),
            "QuoteVolume": np.full(20, 100_000.0),
            "BB_MID": np.full(20, 98.0),
            "MA25": np.full(20, 98.0),
            "MA99": np.full(20, 98.0),
        },
        index=idx,
    )

    hit = sq_radar.RadarHit(
        symbol="TESTUSDT", interval="15m", mode="VWAP_COIL_BREAK", bar_idx=5,
        timestamp=idx[5], price=100.0, stop_price=95.0, breakout_level=99.5, vwap=99.0,
        quote_vol_usd=500_000.0, qv_mult=8.0, vol_accel=2.0, taker_buy_pct=60.0,
        bar_gain_pct=3.0, clv=0.8, bbw_min10_pct=2.0, bbw_max24_pct=4.0,
        rng24_pct=3.0, rng96_pct=5.0, score=85, max_gain_after_pct=32.0, note="test",
    )
    tr_sq = bt_sq.simulate_trade(df, hit)
    assert tr_sq is not None
    assert tr_sq.exit_idx_trend == 6
    assert tr_sq.exit_reason_trend == "保本止盈"

    sig_brk = brk_scanner.BreakoutSignal(
        symbol="TESTUSDT", interval="15m", signal_type="BOTTOM_BREAKOUT", bar_idx=5,
        timestamp=idx[5], entry_price=100.0, stop_price=95.0, target_2r=110.0, target_4r=120.0,
        vol_ratio=12.0, bar_gain_pct=4.0, ma_spread_pct=2.0, base_range_pct=4.0,
        clv=0.8, max_gain_after_pct=32.0, max_r_after=6.4, note="test",
    )
    tr_a = bt_90.simulate_trade(df, sig_brk)
    assert tr_a is not None
    assert tr_a.exit_idx == 6
    assert tr_a.exit_reason_ma25 == "保本止盈"

    tr_b = bt_90.simulate_coiled_trade(df, sig_brk)
    assert tr_b is not None
    assert tr_b.exit_idx == 6
    assert tr_b.exit_reason_ma25 == "保本鎖利止盈"

    sig_ma = ma200_scanner.MA200CoilSignal(
        symbol="TESTUSDT", interval="15m", bar_idx=5, cross_idx=5, timestamp=idx[5],
        entry_price=100.0, stop_price=95.0, ma200=99.0, ext200_pct=1.0, spread8_pct=1.0,
        spread_long_pct=0.8, rng48_pct=3.0, qvm=5.0, vol_acc=2.0, vol_ratio=5.0,
        taker_buy_pct=60.0, clv=0.8, rsi6=70.0, bar_gain_pct=2.0, below96_pct=90.0,
        risk_pct=5.0, score=85, max_gain_after_pct=32.0, note="test",
    )
    sim_ma = ma200_scanner.simulate_ma99_trade(df, sig_ma)
    assert sim_ma is not None
    assert sim_ma["exit_idx"] == 6
    assert sim_ma["reason"] == "保本鎖利"


def test_honest_portfolio_liquidation_liquidity_cap_and_concurrency() -> None:
    t0 = datetime(2026, 5, 1, 0, 0, tzinfo=TPE)
    # Trade 1: huge +200% microcap with tiny $20,000 bar volume -> 2% liquidity cap limits notional to $400!
    tr1 = bt_utils.PortfolioTradeInput(
        symbol="MICROUSDT",
        entry_time=t0,
        exit_time=t0 + timedelta(hours=4),
        entry_price=1.0,
        stop_price=0.90,
        exit_price=3.0,
        risk_pct=10.0,
        net_pnl_pct=200.0,
        r_mult=20.0,
        worst_low_price=0.95,
        bar_quote_vol_usd=20_000.0,
        trailing_24h_qv_usd=5_000_000.0,
    )
    # Trade 2: gap-down crash piercing 3x liquidation threshold (-35% worst low)
    tr2 = bt_utils.PortfolioTradeInput(
        symbol="CRASHUSDT",
        entry_time=t0 + timedelta(hours=5),
        exit_time=t0 + timedelta(hours=6),
        entry_price=100.0,
        stop_price=90.0,
        exit_price=65.0,
        risk_pct=10.0,
        net_pnl_pct=-35.12,
        r_mult=-3.512,
        worst_low_price=65.0,
        bar_quote_vol_usd=50_000_000.0,
        trailing_24h_qv_usd=500_000_000.0,
    )
    port = bt_utils.simulate_honest_portfolio([tr1, tr2], initial_cap=1000.0, max_positions=2, slots_3x=1)

    # On tr1, $400 liquidity cap * +200% = +$800 gain on 3x (equity $1,800 instead of uncapped $7,000)
    # On tr2, uncapped 3x position hits maintenance-margin liquidation -> wiped out to $0!
    assert port["comp_3x"]["liquidity_capped_count"] == 1
    assert port["comp_3x"]["liquidated_count"] == 1
    assert port["comp_3x"]["final_eq"] == pytest.approx(0.0, abs=1e-6)

    # Meanwhile 2% fixed-fractional risk survives the crash cleanly!
    assert port["headline_risk"]["final_eq"] > 1000.0


def test_cli_entrypoints_and_oos_walkforward_smoke(tmp_path: Path) -> None:
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=TPE)
    trades = [
        bt_utils.PortfolioTradeInput(
            symbol="BTCUSDT",
            entry_time=t0 + timedelta(days=10 * k),
            exit_time=t0 + timedelta(days=10 * k, hours=8),
            entry_price=100.0,
            stop_price=96.0,
            exit_price=108.0 if k % 2 == 0 else 96.0,
            risk_pct=4.0,
            net_pnl_pct=7.85 if k % 2 == 0 else -4.15,
            r_mult=1.96 if k % 2 == 0 else -1.04,
            worst_low_price=98.0 if k % 2 == 0 else 96.0,
            bar_quote_vol_usd=10_000_000.0,
            trailing_24h_qv_usd=200_000_000.0,
        )
        for k in range(10)
    ]
    oos = bt_utils.compute_oos_and_walkforward(
        trades,
        window_start=t0,
        window_end=t0 + timedelta(days=100),
        oos_ratio=0.30,
        wf_folds=4,
    )
    assert oos["in_sample"]["n"] + oos["out_of_sample"]["n"] == 10
    assert oos["in_sample"]["n"] > 0 and oos["out_of_sample"]["n"] > 0
    assert len(oos["folds"]) == 4
    html_block = bt_utils.render_oos_walkforward_html(oos)
    assert "In-Sample" in html_block and "Out-of-Sample" in html_block

    for script in (
        "squeeze_breakout_radar.py",
        "binance_breakout_scanner.py",
        "ma200_coil_scanner.py",
        "backtest_180d_squeeze.py",
        "backtest_90d_breakout.py",
        "backtest_365d_ma200.py",
    ):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] / script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{script} --help failed: {proc.stderr}"


def test_end_to_end_backtest_reports_from_local_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = 600
    idx = pd.date_range("2026-09-01", periods=n, freq="15min", tz=TPE)

    # Plant a clean M2 ultra-squeeze climax breakout at bar 320 + follow-through rally
    c = np.full(n, 10.0)
    o = np.full(n, 10.0)
    h = np.full(n, 10.03)
    l = np.full(n, 9.97)
    v = np.full(n, 5_000.0)
    qv = np.full(n, 50_000.0)
    tbv = np.full(n, 2_600.0)

    # Bar 320: tight 6h squeeze (<2.2% BBW, <3.0% rng24) followed by >=30x volume climax, CLV>=0.90, TBR>=0.54
    o[320] = 10.00
    l[320] = 9.99
    h[320] = 10.55
    c[320] = 10.52
    v[320] = 220_000.0
    qv[320] = 2_300_000.0
    tbv[320] = 150_000.0
    for k in range(321, 360):
        step_px = 10.52 + (k - 320) * 0.04
        o[k] = step_px - 0.02
        c[k] = step_px
        h[k] = step_px + 0.03
        l[k] = step_px - 0.03
        v[k] = 20_000.0
        qv[k] = 210_000.0
        tbv[k] = 11_500.0
    for k in range(360, n):
        o[k] = 11.4
        c[k] = 11.3
        h[k] = 11.45
        l[k] = 11.25
        v[k] = 10_000.0
        qv[k] = 113_000.0
        tbv[k] = 5_000.0

    df = pd.DataFrame(
        {
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v,
            "QuoteVol": qv,
            "QuoteVolume": qv,
            "TakerBuyVol": tbv,
        },
        index=idx,
    )
    df.to_pickle(cache_dir / "SOLUSDT_15m_30d.pkl")

    repo_root = Path(__file__).resolve().parents[1]
    out_sq = tmp_path / "out_sq"
    p_sq = subprocess.run(
        [
            sys.executable,
            str(repo_root / "backtest_180d_squeeze.py"),
            "--days", "30",
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_sq),
            "--min-vol", "1000000",
            "--no-fetch-funding",
            "--skip-cards",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_sq.returncode == 0, p_sq.stderr
    sq_html = (out_sq / "index.html").read_text(encoding="utf-8")
    assert "以損定倉複利" in sq_html
    assert "Out-of-Sample" in sq_html
    assert "+0.55R" in sq_html
    assert (out_sq / "equity_curve_30d.png").is_file()

    out_90 = tmp_path / "out_90"
    p_90 = subprocess.run(
        [
            sys.executable,
            str(repo_root / "backtest_90d_breakout.py"),
            "--days", "30",
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_90),
            "--min-vol", "1000000",
            "--no-fetch-funding",
            "--skip-cards",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_90.returncode == 0, p_90.stderr
    b90_html = (out_90 / "index.html").read_text(encoding="utf-8")
    assert "以損定倉複利" in b90_html
    assert "Out-of-Sample" in b90_html

    out_ma = tmp_path / "out_ma"
    p_ma = subprocess.run(
        [
            sys.executable,
            str(repo_root / "backtest_365d_ma200.py"),
            "--cache-dir", str(cache_dir),
            "--out-dir", str(out_ma),
            "--min-vol", "1000000",
            "--no-fetch-funding",
            "--skip-cards",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_ma.returncode == 0, p_ma.stderr
    ma_html = (out_ma / "index.html").read_text(encoding="utf-8")
    assert "以損定倉" in ma_html
    assert "Out-of-Sample" in ma_html
    assert (out_ma / "equity_curve_365d.png").is_file()

