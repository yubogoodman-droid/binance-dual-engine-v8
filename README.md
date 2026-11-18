# 幣安 USDT-M 永續合約 15m 突破訊號掃描器與誠實回測看板生成器

純訊號偵測與歷史回測視覺化工具（**無下單功能、不需 API Key**，全數連線幣安官方 `https://fapi.binance.com` 公開行情端點）。

## 核心設計與防偏誤機制

1. **即時掃描僅針對「已收盤 K 線」觸發**：
   - 所有即時掃描器（`binance_breakout_scanner.py`、`squeeze_breakout_radar.py`、`ma200_coil_scanner.py`、`radar.html`）預設皆為 `drop_forming=True`，自動剔除尚未收盤的當根 K 線以消除盤中重繪（Repainting）；僅在明確傳入 `--include-forming` 時才納入盤中預警。
   - `--watch` 輪詢模式會將已推播訊號持久化至磁碟（預設 `breakout_output/.watch_seen_*.json`，可由 `--seen-file` 指定），重啟腳本不會重複響鈴推播舊訊號。
2. **無偏誤動態歷史標的池（消除倖存者與字母序截斷偏誤）**：
   - 回測不再寫死固定檔數（如舊版文檔之 174 檔 vs 腳本預設 40/80 檔不一致），亦移除強制塞入特定暴漲標的與按檔名字母序截斷之邏輯。
   - 支援載入 `--cache-dir` 內全數純加密貨幣 USDT 永續合約（排除穩定幣、TradFi 美股/大宗商品與槓桿反向代幣），或由 `/fapi/v1/exchangeInfo` 納入回測區間內曾存續之合約，並於**每筆訊號觸發當下**依「前 24h 滾動成交額」（`--min-vol` 與選用 `--top`）進行無前視過濾與排序。
3. **誠實資金曲線、強平/流動性限制與完整交易成本**：
   - **主指標（Headline）**：採用支援多倉併發（預設 `--max-positions 3`）的 **2% 固定風險比例複利（Fixed-Fractional Risk）**。
   - **1x 真實複利 vs 算術總和**：同時提供 1x 無槓桿真實複利淨值，並將單筆報酬相加明確標示為「單筆算術總和（非複利）」。
   - **3x 理論壓力測試**：納入分級維持保證金強平模型（Maintenance Margin Liquidation）與單筆流動性上限（預設不超過突破 K 線成交額 2% / 前 24h 成交額 0.5%），並降級為理論高風險對照。
   - **交易成本與保守盤中出場**：除雙邊 `0.12%` 手續費與滑價外，額外扣除 `/fapi/v1/fundingRate` 歷史資金費率（離線時自動套用可設定之保守費率 `--funding-rate-8h 0.03`，即每 8h `0.03%`）；且於上移保本止損之同一根 K 線保守檢查當根低點是否打穿新止損。
4. **參數凍結樣本外盲測（IS / OOS）與 Walk-Forward 切片**：
   - 所有回測報表皆內建參數凍結之樣本內（In-Sample）與樣本外（Out-of-Sample，`--oos-date` 或 `--oos-ratio`）獨立績效表及滾動時間切片（`--wf-folds`）。

## 檔案總覽

- **即時掃描器**
  - `binance_breakout_scanner.py`：v8 雙引擎掃描器（引擎 A 平地 8 線糾結爆量突破 + 引擎 B 水下穿 `MA200` 二拍確認；預設 `--top 80`）
  - `squeeze_breakout_radar.py`：Squeeze × `12h VWAP` 護盤 × 主動買盤雙模雷達（預設 `--max-symbols 260`）
  - `ma200_coil_scanner.py`：`MA200` 附近 8 線粘合起漲掃描器（預設 `--top 200`）
  - `radar.html`：瀏覽器端即時掃描介面
- **回測與 HTML 看板生成器**
  - `backtest_utils.py`：時點滾動成交額索引、歷史資金費率快取、誠實多槽位資金/強平/流動性模型、IS/OOS 與 Walk-Forward 統計模組
  - `backtest_90d_breakout.py`：v8 雙引擎回測與 HTML 看板生成器
  - `backtest_180d_squeeze.py`：Squeeze × VWAP 雙模雷達回測與 HTML 看板生成器（`+0.55R` 保本鎖定 + `BB` 中軌止盈）
  - `backtest_365d_ma200.py`：`MA200` 粘合策略回測與 IS/OOS 對比看板生成器

## 常用指令

```bash
# 1. 即時全市場掃描（僅掃描已收盤 K 線）與持續盯盤（自動持久化已通知紀錄）
python3 binance_breakout_scanner.py --scan --top 120
python3 squeeze_breakout_radar.py --watch 60 --max-symbols 260
python3 ma200_coil_scanner.py --watch 60 --top 200

# 2. 重新產出誠實回測 HTML 看板（支援本地快取目錄、資金費率與 OOS 切分）
python3 backtest_90d_breakout.py --days 90 --cache-dir breakout_output/cache_365d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4 --out-dir breakout_output/backtest_90d
python3 backtest_90d_breakout.py --days 365 --cache-dir breakout_output/cache_365d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4 --out-dir breakout_output/backtest_365d
python3 backtest_180d_squeeze.py --days 180 --cache-dir breakout_output/cache_180d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4
python3 backtest_180d_squeeze.py --days 365 --cache-dir breakout_output/cache_365d --min-vol 2000000 --oos-ratio 0.30 --wf-folds 4
python3 backtest_365d_ma200.py --cache-dir breakout_output/cache_365d --min-vol 1000000 --oos-date 2026-05-15 --wf-folds 4
```

> **靜態看板說明**：`breakout_output/` 目錄下既有之靜態 HTML 圖卡快照產生於舊版回測邏輯修正前；在可連線 `fapi.binance.com` 或備有本地 `.pkl`/`.npz` 快取之環境執行上述指令，即可覆寫產出含 2% 定倉主指標、資金費率成本與 IS/OOS 盲測表的最新報表。
