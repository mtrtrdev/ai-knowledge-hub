# 💡 AI ナレッジハブ

AI ナレッジハブは、ユーザーのどんな質問にも多角的に分析し、最適な回答を生成する汎用マルチエージェントQ&Aアプリケーションです。Streamlit をフロントエンドに、Google Gemini API をバックエンドのAIエンジンとして利用しています。

## 🚀 概要

このアプリケーションは、従来の単一AIによる回答生成とは異なり、ユーザーの質問を複数の「観点」に分解し、それぞれの観点に特化したAIエージェントが回答を生成します。最終的に、これらの個別回答を「統括AI」がまとめ上げ、詳細で網羅的な最終回答を提供します。

### 主な機能

-   **汎用的な質問入力**: 観光、技術、学習など、幅広いテーマの質問に対応します。
-   **質問テンプレート**: よくある質問や始め方のヒントとして、事前に定義されたテンプレートを選択して質問入力欄に反映できます。
-   **動的な観点生成**: 入力された質問の内容に基づいて、回答に必要な複数の観点（例: 定義、メリット、デメリット、具体例など）をAIが自動で生成します。
-   **マルチエージェントによる回答**:
    -   **観点生成エージェント**: 質問から最適な分析観点を特定します。
    -   **汎用エージェント**: 特定された各観点について、個別にGemini AIを呼び出し、詳細な情報を生成します。
    -   **統括エージェント**: 各エージェントが生成した情報を統合し、論理的で分かりやすい最終回答をまとめます。
-   **過去の質問履歴**: 過去に質問した内容と最終回答をJSON形式で保存し、サイドバーにタイムスタンプと質問文を表示します。クリックすることで、その質問内容を再度入力欄に反映できます。
-   **履歴のリセット**: アプリケーション起動時に、過去の質問履歴が自動的にクリアされます。
-   **ユーザーフレンドリーなUI**: Streamlit の `st.status` や `st.expander` を活用し、処理の進行状況を分かりやすく表示し、途中経過の回答はアコーディオン形式で格納することで、メインコンテンツをすっきりと保ちます。

## 🛠️ 技術スタック

-   **Python**: アプリケーションの主要言語
-   **Streamlit**: 直感的でインタラクティブなWebアプリケーションフレームワーク
-   **Google Gemini API**: AIによるテキスト生成の中核エンジン

## 📂 プロジェクト構造

```
.
├── main.py # Streamlit UI、全体の制御、履歴管理
├── agent.py # AIエージェント（観点生成、汎用回答、統括回答）のロジック
├── requirements.txt # 必要なPythonライブラリのリスト
└── question_history.json # 質問と回答の履歴を保存するJSONファイル（実行時に自動生成/リセット）
```

## ⚙️ セットアップと実行

### 1. 必要なライブラリのインストール

プロジェクトのルートディレクトリで以下のコマンドを実行し、必要なPythonライブラリをインストールします。

```bash
pip install -r requirements.txt
```

`requirements.txt` の内容:

streamlit
google-generativeai

### 2. Google Gemini APIキーの設定

Google Gemini API を利用するためには、APIキーが必要です。以下のいずれかの方法で環境変数 `GEMINI_API_KEY` に設定してください。

-   **Windows (PowerShell)**:
    ```powershell
    $env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    streamlit run main.py
    ```
-   **macOS/Linux (Bash/Zsh)**:
    ```bash
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    streamlit run main.py
    ```
    `YOUR_GEMINI_API_KEY` の部分には、[Google AI Studio](https://aistudio.google.com/) などで取得した実際のAPIキーを記述してください。

### 3. アプリケーションの実行

APIキーを設定した後、プロジェクトのルートディレクトリで以下のコマンドを実行します。

```bash
streamlit run main.py
```

これにより、Webブラウザでアプリケーションが起動します。

## 💡 アプリケーションのフロー

AIナレッジハブの質問から回答までのワークフローは以下の通りです。


```mermaid
graph TD
    subgraph Frontend_main_py
        A[ユーザー入力: 質問テキストエリア] --> B{テンプレート/履歴ボタンクリック}
        B --> C[質問入力欄]
        C --> D[回答生成ボタンクリック]
    end

    subgraph Backend_agent_py
        D --> E[観点生成エージェント: perspective_generation_agent]
        E --> F[複数の汎用エージェント: generic_perspective_agent]
        F --> G[汎用エージェントインスタンス]
        G --> H{回答収集}
        H --> I[統括エージェント: overall_agent]
        I --> J[回答表示]
    end

    subgraph External
        E --> K[Google Gemini AI]
        G --> K
        I --> K
    end

    J --> L[question_history.json: 履歴保存]
    L --> M[サイドバー履歴ボタン]
    M --> B

     style A fill:#ffddaa,stroke:#333,stroke-width:2px,color:#000000
    style B fill:#aaffaa,stroke:#333,stroke-width:2px,color:#000000
    style C fill:#ffddaa,stroke:#333,stroke-width:2px,color:#000000
    style D fill:#ffddaa,stroke:#333,stroke-width:2px,color:#000000
    style E fill:#ddbbff,stroke:#333,stroke-width:2px,color:#000000
    style F fill:#ddbbff,stroke:#333,stroke-width:2px,color:#000000
    style G fill:#ddbbff,stroke:#333,stroke-width:2px,color:#000000
    style H fill:#ddbbff,stroke:#333,stroke-width:2px,color:#000000
    style I fill:#ddbbff,stroke:#333,stroke-width:2px,color:#000000
    style J fill:#aaffee,stroke:#333,stroke-width:2px,color:#000000
    style K fill:#ffeeaa,stroke:#333,stroke-width:2px,color:#000000
    style L fill:#aaddff,stroke:#333,stroke-width:2px,color:#000000
    style M fill:#aaffaa,stroke:#333,stroke-width:2px,color:#000000
