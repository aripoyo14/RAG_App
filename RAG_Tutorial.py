
import streamlit as st
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from google.cloud import vision
import hashlib
from supabase import create_client, Client

# --- requirements.txt ---
# streamlit
# openai
# python-dotenv
# sentence-transformers
# numpy
# PyMuPDF
# google-cloud-vision
# supabase
# ------------------------

# .envファイルから環境変数を読み込む
load_dotenv()

# 学習用のサンプルテキストを外部ファイルからインポート
from sample_texts import sample_text_A, sample_text_B, DISPLAY_NAME_A, DISPLAY_NAME_B

# Google Cloud Vision APIクライアントの初期化（オプション）
vision_client = None
google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_credentials_path and os.path.exists(google_credentials_path):
    try:
        vision_client = vision.ImageAnnotatorClient()
    except Exception as e:
        st.sidebar.warning(f"Google Cloud Vision APIの初期化に失敗しました: {e}")

# Supabaseクライアントの初期化（オプション）
supabase_client = None
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    try:
        supabase_client = create_client(supabase_url, supabase_key)
    except Exception as e:
        st.sidebar.warning(f"Supabaseクライアントの初期化に失敗しました: {e}")
else:
    st.sidebar.warning("Supabaseを使用するには、`.env`ファイルに`SUPABASE_URL`と`SUPABASE_KEY`を設定してください。")

# PDFからテキストを抽出する関数（OCR対応、キャッシュ機能付き）
def extract_text_from_pdf(pdf_file):
    """
    PDFファイルからテキストを抽出する。
    テキストが抽出できない（少ない）場合は、OCR（Google Cloud Vision）を試みる。
    同じPDFファイルの場合は、キャッシュから結果を返す。
    """
    # PDFファイルのハッシュを計算（キャッシュキーとして使用）
    pdf_bytes = pdf_file.read()
    pdf_file.seek(0)  # ファイルポインタをリセット
    
    # ファイル名と内容のハッシュを組み合わせてキーを生成
    file_hash = hashlib.md5(pdf_bytes).hexdigest()
    file_name = pdf_file.name if hasattr(pdf_file, 'name') else 'unknown'
    cache_key = f"{file_name}_{file_hash}"
    
    # キャッシュの初期化
    if 'pdf_cache' not in st.session_state:
        st.session_state.pdf_cache = {}
    
    # キャッシュに結果がある場合はそれを返す
    if cache_key in st.session_state.pdf_cache:
        # st.info(f"キャッシュからPDF「{file_name}」のテキストを取得しました。")
        return st.session_state.pdf_cache[cache_key]
    
    # 1. PyMuPDFでテキスト抽出を試みる
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
        
        doc.close()
        
        # テキストが実際に抽出できたか判定（空白や改行のみでないかチェック）
        # 意味のある文字（空白・改行・タブ以外）が含まれているか確認
        text_without_whitespace = ''.join(full_text.split())
        if len(text_without_whitespace) > 0:
            # テキスト情報が含まれている場合は、OCR処理をスキップしてテキストを返す
            st.success("PDFからテキスト情報を取得しました。OCR処理はスキップします。")
            # キャッシュに保存
            st.session_state.pdf_cache[cache_key] = full_text
            return full_text
        
        # テキストが抽出できない（空白のみ）場合はOCR処理に移行
        st.info("PDFにテキスト情報が含まれていません。OCR処理を実行します...")
        
    except Exception as e:
        st.warning(f"PyMuPDFでのテキスト抽出エラー: {e}。OCR処理に移行します。")

    # 2. OCR処理 (テキスト抽出失敗時またはテキストが少ない場合)
    if vision_client is None:
        st.error("OCR機能を使用するには、`.env`ファイルに`GOOGLE_APPLICATION_CREDENTIALS`を設定してください。")
        return None
    
    full_text_ocr = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            
            # ページを画像(PNG)に変換
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            
            image = vision.Image(content=img_bytes)
            
            # Vision APIでOCR実行
            response = vision_client.document_text_detection(image=image)
            
            if response.full_text_annotation:
                full_text_ocr += response.full_text_annotation.text + "\n\n"
            
            # プログレスバーを更新
            progress = (page_num + 1) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"OCR処理中: ページ {page_num + 1}/{total_pages}")
        
        doc.close()
        progress_bar.empty()
        status_text.empty()
        
        if hasattr(response, 'error') and response.error.message:
            raise Exception(f"Vision APIエラー: {response.error.message}")
        
        # キャッシュに保存
        st.session_state.pdf_cache[cache_key] = full_text_ocr
        st.success("Google Cloud VisionによるOCR処理に成功しました。")
        return full_text_ocr

    except Exception as e:
        st.error(f"OCR処理中にエラーが発生しました: {e}")
        if 'doc' in locals():
            doc.close()
        return None

# チャンクをベクトル化してSupabaseに登録する関数
def process_and_upload_to_supabase(chunks, file_metadata=None):
    """
    分割実行で生成されたチャンクをベクトル化し、Supabaseに登録する。
    
    Args:
        chunks: 登録するチャンクのリスト
        file_metadata: ファイルのメタデータ（辞書形式）
    """
    if supabase_client is None:
        st.error("Supabaseクライアントが初期化されていません。`.env`ファイルに`SUPABASE_URL`と`SUPABASE_KEY`を設定してください。")
        return False
    
    if not chunks or len(chunks) == 0:
        st.error("チャンクが空です。先に「分割実行」ボタンを押してチャンクを生成してください。")
        return False
    
    try:
        # 1. ベクトル化（Embedding）
        with st.spinner("チャンクをベクトル化中..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(chunks)
            st.info(f"ベクトル化完了。チャンク数: {len(chunks)}, ベクトル形状: {embeddings.shape}")
        
        # 2. Supabaseに登録
        with st.spinner("Supabaseへのデータ登録中..."):
            data_to_upload = []
            default_metadata = file_metadata if file_metadata else {"source_file": "unknown"}
            
            for i, chunk in enumerate(chunks):
                data_to_upload.append({
                    "content": chunk,
                    "embedding": embeddings[i].tolist(),  # ベクトルをリスト形式に変換
                    "metadata": default_metadata
                })
            
            # upsertでデータを挿入（バッチ処理）
            response = supabase_client.table("documents").upsert(data_to_upload).execute()
            
            # レスポンスの確認
            if hasattr(response, 'data') and response.data:
                st.success(f"Supabaseへの登録が完了しました。（{len(response.data)}件）")
                return True
            else:
                st.error("Supabaseへの登録に失敗しました。レスポンスにデータが含まれていません。")
                return False
                
    except Exception as e:
        st.error(f"Supabaseへの登録中にエラーが発生しました: {e}")
        return False

# --- App Title and Description ---
st.set_page_config(page_title="RAGステップ・バイ・ステップ学習", layout="wide")
st.title("触って学ぶ！RAG（Retrieval-Augmented Generation）の仕組み")
st.write("""
このアプリは、RAGの3つの主要なステップ（チャンキング、検索、生成）を、実際に操作しながら視覚的に理解するための学習ツールです。
各ステップで何が行われているのか、その「中身」を覗いてみましょう。
""")

# --- API Key Check ---
# OpenAI APIキーが環境変数に設定されているか確認
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key
    st.sidebar.success("OpenAI APIキーが設定されました。")
else:
    st.sidebar.error("`.env`ファイルに`OPENAI_API_KEY`を設定してください。")

# --- Step A: Chunking ---
st.header("ステップA: チャンキング（テキストの分割）")
st.write("最初のステップは、元の大きなテキストを、AIが扱いやすい小さな「チャンク」に分割することです。")

# テキストソースの選択（PDFまたはサンプルテキスト）
text_source_option = st.radio(
    "テキストソースを選択:",
    options=["サンプルテキスト", "PDFファイル"],
    horizontal=True,
    key="text_source_radio"
)

source_text = None
text_source_type = None
uploaded_file = None
selected_sample = None

if text_source_option == "PDFファイル":
    # PDFアップロード機能
    uploaded_file = st.file_uploader(
        "PDFファイルをアップロード:",
        type=['pdf'],
        help="PDFファイルをアップロードすると、そのテキストがチャンキングのソースとして使用されます。"
    )
    
    if uploaded_file is not None:
        # PDFファイルのハッシュを計算してキャッシュをチェック
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # ファイルポインタをリセット
        file_hash = hashlib.md5(pdf_bytes).hexdigest()
        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'unknown'
        cache_key = f"{file_name}_{file_hash}"
        
        # ファイル情報をsession_stateに保存
        st.session_state.current_file_name = file_name
        
        # キャッシュの初期化
        if 'pdf_cache' not in st.session_state:
            st.session_state.pdf_cache = {}
        
        # キャッシュに結果がある場合はそれを使用、ない場合は抽出を実行
        if cache_key in st.session_state.pdf_cache:
            source_text = st.session_state.pdf_cache[cache_key]
            text_source_type = "PDF"
            st.session_state.text_source_type = "PDF"
            # st.info(f"キャッシュからPDF「{file_name}」のテキストを取得しました。")
        else:
            # PDFからテキストを抽出
            with st.spinner("PDFからテキストを抽出中..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    source_text = extracted_text
                    text_source_type = "PDF"
                    st.session_state.text_source_type = "PDF"
                    st.success(f"PDFファイル「{file_name}」からテキストを抽出しました。")
                else:
                    st.error("PDFからテキストを抽出できませんでした。")
                    source_text = None
    else:
        st.info("PDFファイルをアップロードしてください。")
        # session_stateからクリア
        if 'current_file_name' in st.session_state:
            del st.session_state.current_file_name
else:
    # サンプルテキストを選択
    sample_texts = {
        DISPLAY_NAME_A: sample_text_A,
        DISPLAY_NAME_B: sample_text_B
    }
    
    selected_sample = st.selectbox(
        "1. 学習用の原文を選択してください:",
        options=list(sample_texts.keys())
    )
    source_text = sample_texts[selected_sample]
    text_source_type = "サンプルテキスト"
    st.session_state.text_source_type = "サンプルテキスト"
    # 選択したサンプルをsession_stateに保存
    st.session_state.current_sample_name = selected_sample

if source_text:
    with st.expander(f"選択した原文を表示（{text_source_type}）"):
        st.text(source_text)

split_method = st.radio(
    "2. 分割方法を選択してください:",
    options=["固定文字数", "改行（\\n）", "句読点（。）"],
    horizontal=True
)

chunk_size = 50
if split_method == "固定文字数":
    chunk_size = st.slider("チャンクサイズ（文字数）:", min_value=50, max_value=500, value=150, step=10)

chunks = []
if st.button("分割実行", key="chunking_button"):
    if source_text is None:
        st.error("テキストが選択されていません。サンプルテキストを選択するか、PDFファイルをアップロードしてください。")
    else:
        # 分割方法をsession_stateに保存
        st.session_state.split_method = split_method
        if split_method == "固定文字数":
            chunks = [source_text[i:i+chunk_size] for i in range(0, len(source_text), chunk_size)]
            st.session_state.chunk_size = chunk_size
        elif split_method == "改行（\\n）":
            chunks = [p for p in source_text.split('\n') if p.strip()]
        elif split_method == "句読点（。）":
            chunks = [p + "。" for p in source_text.split('。') if p.strip()]

if 'chunks' not in st.session_state:
    st.session_state.chunks = []

if chunks:
    st.session_state.chunks = chunks

if st.session_state.chunks:
    st.metric("合計チャンク数", len(st.session_state.chunks))
    st.write("---")
    st.subheader("分割されたチャンク一覧")
    for i, chunk in enumerate(st.session_state.chunks):
        with st.container(border=True):
            st.write(f"**チャンク {i+1}**")
            st.text(chunk)
    
    # Supabaseへの登録機能
    if supabase_client:
        st.write("---")
        st.subheader("Supabaseへの登録")
        st.write("「分割実行」で生成されたチャンクをベクトル化してSupabaseに登録できます。")
        
        # メタデータを準備
        current_text_source_type = st.session_state.get('text_source_type', 'unknown')
        current_split_method = st.session_state.get('split_method', 'unknown')
        if current_text_source_type == "PDF":
            # PDFファイルの場合
            file_name = st.session_state.get('current_file_name', 'unknown')
            metadata = {
                "source_file": file_name,
                "type": "pdf",
                "chunking_method": current_split_method
            }
            # 固定文字数の場合はチャンクサイズも保存
            if current_split_method == "固定文字数":
                metadata["chunk_size"] = st.session_state.get('chunk_size', 150)
        else:
            # サンプルテキストの場合
            sample_name = st.session_state.get('current_sample_name', 'sample_text')
            metadata = {
                "source_file": sample_name,
                "type": "sample_text",
                "chunking_method": current_split_method
            }
            # 固定文字数の場合はチャンクサイズも保存
            if current_split_method == "固定文字数":
                metadata["chunk_size"] = st.session_state.get('chunk_size', 150)
        
        if st.button("Supabaseに登録", key="upload_to_supabase"):
            process_and_upload_to_supabase(st.session_state.chunks, file_metadata=metadata)

# Supabaseでベクトル検索を実行する関数
def search_documents_supabase(query_text, threshold=0.5, count=5):
    """
    テキストクエリをベクトル化し、Supabaseでベクトル検索を実行する
    
    Args:
        query_text: 検索クエリのテキスト
        threshold: 類似度の閾値（デフォルト: 0.5）
        count: 返す検索結果の数（デフォルト: 5）
    
    Returns:
        検索結果のリスト（chunk, scoreのタプルのリスト）
    """
    if supabase_client is None:
        st.error("Supabaseクライアントが初期化されていません。`.env`ファイルに`SUPABASE_URL`と`SUPABASE_KEY`を設定してください。")
        return None
    
    try:
        # 1. 質問をベクトル化
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query_text])[0].tolist()
        
        # 2. SupabaseのRPC (Remote Procedure Call) でSQL関数を呼び出す
        response = supabase_client.rpc('match_documents', {
            'query_embedding': query_embedding,
            'match_threshold': threshold,
            'match_count': count
        }).execute()
        
        if response.data:
            # (chunk, score)のタプルのリストに変換
            scored_chunks = [(doc['content'], doc['similarity']) for doc in response.data]
            return scored_chunks
        else:
            return []
            
    except Exception as e:
        st.error(f"Supabaseでの検索エラー: {e}")
        return None

# --- Step B: Retrieval ---
st.header("ステップB: 検索（Retrieval）")
st.write("次に、ユーザーの質問と意味が近いチャンクを、大量のチャンクの中から探し出します。この「検索」がRAGの核となる部分です。")

if st.session_state.chunks:
    question = st.text_input("3. サンプルテキストに関する質問を入力してください:", placeholder="例：主人公の名前は？")

    # 検索ボタンを2つ配置
    col1, col2 = st.columns(2)
    
    with col1:
        # ローカル検索（既存機能）
        local_search_clicked = st.button("検索実行（ローカル）", key="retrieval_button_local", use_container_width=True)
        
    with col2:
        # Supabase検索（新機能）
        supabase_search_clicked = st.button("検索実行（Supabase）", key="retrieval_button_supabase", use_container_width=True, disabled=supabase_client is None)
        if supabase_client is None:
            st.caption("⚠️ Supabaseが設定されていません")

    # ローカル検索の実行
    if local_search_clicked and question:
        with st.spinner("Embeddingモデルを読み込み、ベクトル化と類似度計算を実行中です..."):
            try:
                # 1. Load model
                model = SentenceTransformer('all-MiniLM-L6-v2')

                # 2. Embed all chunks
                chunk_embeddings = model.encode(st.session_state.chunks)

                # 3. Embed question
                question_embedding = model.encode([question])

                # 4. Calculate cosine similarity
                similarities = np.dot(chunk_embeddings, question_embedding.T) / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) * np.linalg.norm(question_embedding, keepdims=True))
                
                # Create a list of tuples (chunk, score) and sort it
                scored_chunks = sorted(zip(st.session_state.chunks, similarities.flatten()), key=lambda x: x[1], reverse=True)
                
                st.session_state.scored_chunks = scored_chunks
                st.session_state.question = question
                st.session_state.search_method = "local"

            except Exception as e:
                st.error(f"検索実行中にエラーが発生しました: {e}")

    # Supabase検索の実行
    if supabase_search_clicked and question:
        with st.spinner("Supabaseでベクトル検索を実行中です..."):
            try:
                # 検索パラメータの設定
                threshold = st.session_state.get('supabase_search_threshold', 0.5)
                count = st.session_state.get('supabase_search_count', 5)
                
                scored_chunks = search_documents_supabase(question, threshold=threshold, count=count)
                
                if scored_chunks is not None:
                    st.session_state.scored_chunks = scored_chunks
                    st.session_state.question = question
                    st.session_state.search_method = "supabase"
                    
                    if len(scored_chunks) == 0:
                        st.warning(f"類似度 {threshold} 以上で関連するチャンクは見つかりませんでした。")
                    else:
                        st.info(f"類似度 {threshold} 以上で {len(scored_chunks)} 件の関連チャンクが見つかりました。")

            except Exception as e:
                st.error(f"Supabase検索実行中にエラーが発生しました: {e}")

    # Supabase検索のパラメータ設定
    if supabase_client:
        st.write("---")
        with st.expander("Supabase検索の設定", expanded=False):
            st.caption("Supabaseに登録されたデータから検索します。先にSupabaseにデータを登録してください。")
            threshold = st.slider(
                "類似度の閾値:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('supabase_search_threshold', 0.5),
                step=0.05,
                help="この値以上の類似度を持つチャンクのみが検索結果に含まれます。"
            )
            st.session_state.supabase_search_threshold = threshold
            
            count = st.slider(
                "検索結果の最大件数:",
                min_value=1,
                max_value=20,
                value=st.session_state.get('supabase_search_count', 5),
                step=1
            )
            st.session_state.supabase_search_count = count

    # 検索結果の表示
    if 'scored_chunks' in st.session_state and 'question' in st.session_state:
        search_method = st.session_state.get('search_method', 'local')
        method_name = "ローカル" if search_method == "local" else "Supabase"
        
        st.subheader(f"4. 検索結果（{method_name}検索、関連度スコア順）")
        st.write(f"質問と各チャンクの意味的な近さを「関連度スコア」として計算し、スコアの高い順に並べ替えました。（検索方法: {method_name}）")
        for i, (chunk, score) in enumerate(st.session_state.scored_chunks):
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**チャンク {i+1}**")
                    st.text(chunk)
                with col2:
                    st.write(f"**関連度スコア**")
                    st.progress(float(score))
                    st.write(f"{score:.4f}")
else:
    st.info("ステップAでチャンクを生成してください。")


# --- Step C: Generation ---
st.header("ステップC: 生成（Generation）")
st.write("最後に、検索で見つけた関連度の高いチャンクを「参考情報」としてAIに渡し、質問に対する回答を生成させます。")

if 'scored_chunks' in st.session_state and openai_api_key:
    
    max_chunks = len(st.session_state.scored_chunks)
    top_n = st.slider(
        "回答生成に使うチャンクの数を選択してください:",
        min_value=1,
        max_value=min(max_chunks, 10),  # Use all available chunks up to a max of 10
        value=min(max_chunks, 3),      # Default to 3 or the max available if less
        step=1
    )

    context_chunks = [chunk for chunk, score in st.session_state.scored_chunks[:top_n]]
    context_for_prompt = "\n\n".join(context_chunks)

    st.success(f"検索結果の上位{top_n}件を「AIへの参考情報（コンテキスト）」として使用します。")

    # --- Final Prompt Visualization ---
    st.subheader("5. AIに渡す「最終プロンプト」の可視化")
    st.write("これが、RAGを使って回答を生成するために、AI（OpenAI）に渡す実際のプロンプトです。")
    
    final_prompt = f"""
以下の参考情報を使って、質問に答えてください。参考情報に答えがない場合は、あなたの思考過程を示した上で「分かりません」と答えてください。

---
【参考情報】
{context_for_prompt}
---
【質問】
{st.session_state.get('question', '')}
"""
    st.code(final_prompt, language="text")

    # --- Answer Generation ---
    st.subheader("6. 回答の比較")

    if st.button("回答を生成", key="generate_answers"):
        # Reset previous answers
        if 'rag_answer' in st.session_state:
            del st.session_state['rag_answer']
        if 'no_rag_answer' in st.session_state:
            del st.session_state['no_rag_answer']

        with st.spinner("回答生成中..."):
            try:
                # RAGあり
                client_rag = openai.OpenAI()
                response_rag = client_rag.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": final_prompt}
                    ]
                )
                st.session_state.rag_answer = response_rag.choices[0].message.content
            except Exception as e:
                st.error(f"RAGあり回答の生成中にエラーが発生しました: {e}")

            try:
                # RAGなし
                client_no_rag = openai.OpenAI()
                no_rag_prompt = st.session_state.get('question', '')
                response_no_rag = client_no_rag.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": no_rag_prompt}
                    ]
                )
                st.session_state.no_rag_answer = response_no_rag.choices[0].message.content
            except Exception as e:
                st.error(f"RAGなし回答の生成中にエラーが発生しました: {e}")


    # Display answers if they exist in session state
    if 'rag_answer' in st.session_state or 'no_rag_answer' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if 'rag_answer' in st.session_state:
                st.info("**RAGありの回答**\n\n上記最終プロンプトを使用")
                st.write(st.session_state.rag_answer)
        with col2:
            if 'no_rag_answer' in st.session_state:
                st.warning("**RAGなしの回答**\n\n上記の参考情報は参照せず、OpenAIのデフォルトの知識・検索結果を使用")
                st.write(st.session_state.no_rag_answer)


elif not openai_api_key:
    st.warning("ステップCに進むには、`.env`ファイルに`OPENAI_API_KEY`を設定してください。")
else:
    st.info("ステップBで検索を実行してください。")

