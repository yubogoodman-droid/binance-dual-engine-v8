# 幣安永續合約「v8 雙引擎合體版」策略掃描與 174 檔全市場回測看板

結合兩套 0% 訊號重疊的 15m 突破引擎，針對 **174 檔純加密貨幣 USDT 永續合約** 進行無差別實盤級回測（已扣 0.12% 來回手續費與滑價）：

- **★ 引擎 A（平地巨量突破）**：全 8 條均線 (`MA7 ~ MA200`) 糾結度 ≤ 3.4% + 底部黃金收斂 / 極窄箱體 + 10x~34x 爆量突破 → 沿 15m `MA25` 移動停利
- **🐂 引擎 B（牛來 · 水下穿 MA200 二拍）**：`MA200` 水下築底 ≥ 80 根 + 7 線黏帶 ≤ 1.8% + 第一拍 2~8x 穿 `MA200` + 第二拍紅 K ≥ 1% 強確認 → 沿 `MA99` (`1H MA25`) 移動停利

## 📊 線上戰績看板 (GitHub Pages)

- **[365 天全市場 174 檔戰績看板](https://yubogoodman-droid.github.io/binance-dual-engine-v8/breakout_output/backtest_365d/index.html)**
- **[90 天全市場 174 檔戰績看板](https://yubogoodman-droid.github.io/binance-dual-engine-v8/breakout_output/backtest_90d/index.html)**

## 🛠️ 核心檔案

- `binance_breakout_scanner.py`：雙引擎實時掃描器與訊號偵測核心
- `backtest_90d_breakout.py`：174 檔全市場回測引擎與 HTML 看板生成器
- `breakout_output/`：90 天與 365 天回測 HTML 看板、資金曲線與逐筆 K 線復盤圖卡
