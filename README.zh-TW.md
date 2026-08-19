# Medical Image Agent Harness

這是一套供 Codex 與 GitHub Copilot 共用、可測試且不綁模型供應商的醫學影像
共同判讀科學 harness。它把品質守門、系統性盲讀、證據定位、反證式 second
look、不確定性、來源追溯與評測契約從商品整合中獨立出來。

> 僅供研究與評估，不是醫療器材，不提供自主診斷，也不會自動寫回臨床系統。
> 所有判讀與行動都必須由具資格的人員覆核。

## 核心流程

```text
已去識別來源／study manifest
  → 完整性與影像品質守門
  → 不先看工具答案的系統性觀察
  → 選配獨立模型／工具證據
  → 針對衝突與高風險盲點 second look
  → 綁定原始影像座標的證據驗證
  → 由 atomic observation ledger 重建摘要
  → schema／安全驗證
  → 人工覆核
```

共同方法的唯一來源是
`.agents/skills/medical-image-reading/SKILL.md`；Codex 與 Copilot 的專屬設定都只
是薄 adapter，不各自複製判讀方法。

## 開發與驗證

```bash
uv sync --extra dev
uv run python scripts/check_compatibility.py
uv run pytest
uv run medical-image-harness fingerprint
```

公開 repo 刻意不含螢幕擷取、viewer／overlay、PACS/EHR 寫回、商品 plugin、
憑證、私有模型與權重。私人產品應以固定 commit 的 Git submodule 單向依賴
本 harness，並實作 provider adapter。

`AnalyzerPort` 回傳的是 typed draft；`OutputValidator` 只做相容性正規化，並不
代表 canonical contract 已通過。可信任的 host 必須補上 study manifest、精確
provenance、observation/evidence ledger、workflow events、assessment scope 與人工
覆核狀態，最後呼叫 `AnalysisResult.to_contract_payload()`；任何缺漏都會 fail closed。

詳見 [科學方法](docs/METHODOLOGY.md)、[整合邊界](docs/INTEGRATION.md) 與
[公開 prior art／授權盤點](docs/PRIOR_ART.md)。
