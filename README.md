# LLM-->Pandocでの発表スライド（半）自動生成
[Kaggle LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery/overview)における我々の[12th解法](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494526)紹介スライドを作成する場面を例に.

## Workflow Diagram

```mermaid
graph TD
    H[<span style='color:orange;'>Overview and Our solution</span>] --> A[01_prompts_kaggle_competition.txt]
    A -->|Claude3.5 sonnet| B[01_claude35sonnet_output.md]

    I[<span style='color:orange;'>Top Solutions #1-5</span>] --> C[02_prompts_solusion_summary.txt]
    
    C -->|Claude3.5 sonnet| D[02_claude35sonnet_solution_summary_output.md]
    E[03_prompts_to_combine_markdowns.txt]
    subgraph Outputs
        B
        D
    end
    subgraph Prompt_w_Outputs
        E
        Outputs
    end
    Prompt_w_Outputs -->|Claude3.5 sonnet| F[03_claude35sonnet_merged_presentation_output.md]
    F -->|Pandoc| G[<span style='color:red;'>kaggle_presentation_merged.pptx</span>]
```

## Usage
### 1. Claude3.5 sonnetへのプロンプトをテキストで用意
- ChatGPT4oも試したが，どちらもURLの内容読み込みが甘かったため, べた貼り. (要改善)
- 英語の方が良質な出力を期待できる気がしているのでとりあえず英語 (winner's callとかは英語だし.)
- `01_prompts_kaggle_competition.txt`

### 2. Claude3.5 sonnetから出力`.md`を取得
- `01_prompts_kaggle_competition.txt`　をClaude3.5 sonnetへドラッグ&ドロップ.
- 最初は英語の出力なので, "同じ内容を日本語で出力してください"と指示.
- 今回はなぜか途中で切れたりしたため, 手作業でコピペ. (要改善)
  ![image](https://github.com/Masaaaato/LLM-Pandoc-PPTX/assets/93956588/e00f55f0-4e4e-48b4-8338-12a0d0de47f2)
- `output/01_claude35sonnet_output_ja.md`

### 3. (Optional)スライドテンプレートを用意
- [ここ](https://yyhhyy.hatenablog.com/entry/2019/10/22/100000)を参考にした.
- `output/template.pptx`
- 今回は特にこだわりはない.
    - スライドマスターにて設定.
    - 比16:9, 日本語メイリオ, 英語Arial, スライド番号挿入くらい.
    - コーポレートカラーやデザインがあれば.
- Markdownレベル1は以下について，予めフォントサイズを指定しておくと便利. (フォントサイズ48)
  ![image-2](https://github.com/Masaaaato/LLM-Pandoc-PPTX/assets/93956588/b9271cb9-5773-4f98-ac90-3139f869120a)
- Markdownレベル2は以下について，予めフォントサイズを指定しておくと便利. (タイトル36, 第2レベル18, 行間オプション倍率1.1にした.)
  ![image-1](https://github.com/Masaaaato/LLM-Pandoc-PPTX/assets/93956588/1d736727-f4b0-4e78-a7cb-2817f9594e49)


### 4. Pandoc
- install (私の環境: WSL2 Ubuntu 22.04.3 LTS)
```bash
sudo apt update
sudo apt install pandoc
```
- 以下を実行
```bash
cd output
pandoc -s 01_claude35sonnet_output_ja.md -o kaggle_presentation.pptx --reference-doc=template.pptx
```

### 5. (Optional) Top Solutionのまとめを資料に追加
- 1-5位までの解法をべた貼りした`02_prompts_solusion_summar.txt`を用いて`02_claude35sonnet_solution_summary_output_ja.md`を取得.
- `03_prompts_to_combine_markdowns.txt`の内容をClaude3.5 sonnetに入力の上, `01_claude35sonnet_output_ja.md`と`02_claude35sonnet_solution_summary_output_ja.md`を添付.
- `output/03_claude35sonnet_merged_presentation_output_ja.md`を得る.
- (それぞれの解法を###セクションで区切ると1スライドに乗ってしまうので, 適宜マニュアルで##セクション追加.) (要改善)
- 同様にPPTX生成.
```bash
cd output
pandoc -s 03_claude35sonnet_merged_presentation_output_ja.md -o kaggle_presentation_merged.pptx --reference-doc=template.pptx
```
