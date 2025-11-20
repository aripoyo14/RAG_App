# RAGアプリケーション チュートリアル

本リポジトリでは、RAG（Retrieval-Augmented Generation）の基本的な処理の流れを学ぶことができる学習用アプリケーションを提供しています。

## 📚 このリポジトリについて

RAGは、大規模言語モデル（LLM）に外部の知識ベースを組み合わせる手法です。本リポジトリでは、RAGの3つの主要なステップ（チャンキング、検索、生成）を、実際にコードを書いたり、実行しながら学ぶことができます。

### RAGの3ステップ

1. **チャンキング（Chunking）**: 大きなテキストを小さな断片（チャンク）に分割
2. **検索（Retrieval）**: ユーザーの質問に関連するチャンクを見つけ出す
3. **生成（Generation）**: 検索で見つけた情報を基に、AIが回答を生成

## 📁 ファイル構成

### 学習用ファイル

#### `RAG_Tutorial.ipynb`
- **役割**: Jupyter Notebook形式のチュートリアル資料
- **内容**: RAGの3ステップ（チャンキング、検索、生成）を一つ一つ解説し、コードを実行しながら学習できます

#### `RAG_Tutorial.py`
- **役割**: 完全な回答版のStreamlitアプリケーション
- **内容**: RAGの3ステップを視覚的に操作しながら学べるWebアプリです。各ステップで何が行われているのか、その「中身」を確認できます
- **実行方法**: `streamlit run RAG_Tutorial.py`

### 課題用ファイル

#### `RAG_Tutorial_homework.py`
- **役割**: 課題用のStreamlitアプリケーション（コードの一部が削除されています）
- **内容**: 5箇所のコードが削除されているため、以下を参考に削除された部分を補完してください
  - **Q1**: GPTモデルを選択できるようにするコード（`RAG_Tutorial.py`の42-47行目を参照）
  - **Q2**: 選択した原文を表示するコード（`RAG_Tutorial.py`の65-66行目を参照）
  - **Q3**: 分割方法を選択できるようにするコード（`RAG_Tutorial.py`の68-72行目を参照）
  - **Q4**: コサイン類似度を使用して、チャンクと質問の類似度を計算するコード（`RAG_Tutorial.py`の113-122行目を参照）
  - **Q5**: RAGありとRAGなしの回答を横に並べて表示するコード（`RAG_Tutorial.py`の229行目を参照）
- **参考資料**:
  - `RAG_Tutorial.py`（完全な回答版）
  - `RAG_Tutorial.ipynb`（技術的な解説）
  - [Streamlit公式ドキュメント](https://docs.streamlit.io/)

### 発展版ファイル

#### `RAG_APP_fullmodel.py`
- **役割**: より実用的なRAGアプリケーション（発展版）
- **追加機能**:
  - **SupabaseをベクトルDBとして使用**: ベクトルデータベースとしてSupabaseを活用し、大量のドキュメントを効率的に管理
  - **PDFドキュメントの読み込み**: PDFファイルからテキストを抽出
  - **OCR処理（Google Cloud Vision API）**: テキスト情報が無いPDF（画像スキャンされたドキュメント等）に対してOCR処理を実行
- **用途**: 最終発表のアプリ作成や自主学習に役立ててください
- **必要な環境変数**: 
  - `OPENAI_API_KEY`: OpenAI APIキー（必須）
  - `SUPABASE_URL`: SupabaseプロジェクトURL（オプション）
  - `SUPABASE_KEY`: Supabase APIキー（オプション）
  - `GOOGLE_VISION_API_KEY`: Google Cloud Vision APIキー（OCR機能を使用する場合）

## 🚀 セットアップ方法

### 1. 仮想環境(venv)の作成と有効化（任意）

Pythonの仮想環境を作成して、プロジェクトの依存関係を管理します。

#### Mac / Linux の場合

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate
```

#### Windows の場合

```powershell
# 仮想環境を作成
python -m venv venv

# 仮想環境を有効化
venv\Scripts\activate
```

**注意**: 仮想環境が有効化されると、ターミナルのプロンプトの前に `(venv)` と表示されます。

仮想環境を無効化する場合は、以下のコマンドを実行してください：
- Mac / Linux: `deactivate`
- Windows: `deactivate`

### 2. 必要なライブラリのインストール

仮想環境を有効化した状態で、以下のコマンドを実行してください：

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env`ファイルを作成し、以下の環境変数を設定してください：

```env
OPENAI_API_KEY=Your OpenAI API KEY
```

発展版（`RAG_APP_fullmodel.py`）を使用する場合は、以下の環境変数も追加できます：

```env
SUPABASE_URL=Your Supabase Project URL
SUPABASE_KEY=Your Supabase API Key
GOOGLE_VISION_API_KEY=Your Google Cloud Vision API Key
```

詳細は`env.example`を参照してください。

### 4. アプリケーションの起動

**基本版（課題用）**:
```bash
streamlit run RAG_Tutorial_homework.py
```

**回答版（確認用）**:
```bash
streamlit run RAG_Tutorial.py
```

**発展版**:
```bash
streamlit run RAG_APP_fullmodel.py
```

## 📖 課題の進め方

1. `RAG_Tutorial.ipynb`を開いて、RAGの基本的な仕組みを学習
2. `RAG_Tutorial_homework.py`を開いて、削除された5箇所のコードを補完
3. `RAG_Tutorial.py`を参考にしながら、各質問（Q1-Q5）の回答を記入
4. アプリケーションを実行して、正しく動作することを確認

## 🎯 RAGの精度を向上させる手法

RAGの精度を向上させたい場合は、以下のような手法があります。興味のある方はぜひ挑戦してみてください。

| 難易度 | 手法 | 概要 |
|--------|------|------|
| 低 🔰 | チャンク戦略の最適化 | チャンクサイズとオーバーラップを調整する |
| 低 🔰 | 埋め込みモデルの変更 | より高性能なEmbeddingモデルに入れ替える |
| 中 🧑‍💻 | Re-ranking (リランキング) | 取得したチャンクを、再度並べ替えて絞り込む |
| 中 🧑‍💻 | Hybrid Search (ハイブリッド検索) | ベクトル検索とキーワード検索（BM25等）を組み合わせる |
| 中 🧑‍💻 | Query Expansion (クエリ拡張) | ユーザーの質問をLLMで複数パターンに書き換えて検索する |
| 高 🚀 | Parent Document Retriever | 検索は小さなチャンクで、LLMにはその親（大きな）チャンクを渡す |
| 高 🚀 | Sentence-Window Retrieval | 検索は「文」単位で、LLMにはその「前後」を含めて渡す |

## 📝 補足情報

- サンプルテキストは`sample_texts.py`から読み込まれます
- PDFファイルは`data/`ディレクトリに配置してください（発展版で使用）
- Jupyter Notebookを使用する場合は、`RAG_Tutorial.ipynb`を開いてください
