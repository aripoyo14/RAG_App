import streamlit as st
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# --- requirements.txt ---
# streamlit
# openai
# python-dotenv
# sentence-transformers
# numpy
# ------------------------

# .envファイルから環境変数を読み込む
load_dotenv()

# 学習用のサンプルテキストを外部ファイルからインポート
from sample_texts import sample_text_A, sample_text_B, DISPLAY_NAME_A, DISPLAY_NAME_B

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

# Sample Texts
sample_texts = {
    DISPLAY_NAME_A: sample_text_A,
    DISPLAY_NAME_B: sample_text_B
}

selected_sample = st.selectbox(
    "1. 学習用の原文を選択してください:",
    options=list(sample_texts.keys())
)
source_text = sample_texts[selected_sample]

with st.expander("選択した原文を表示"):
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
    if split_method == "固定文字数":
        chunks = [source_text[i:i+chunk_size] for i in range(0, len(source_text), chunk_size)]
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

# --- Step B: Retrieval ---
st.header("ステップB: 検索（Retrieval）")
st.write("次に、ユーザーの質問と意味が近いチャンクを、大量のチャンクの中から探し出します。この「検索」がRAGの核となる部分です。")

if st.session_state.chunks:
    question = st.text_input("3. サンプルテキストに関する質問を入力してください:", placeholder="例：主人公の名前は？")

    if st.button("検索実行", key="retrieval_button") and question:
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


            except Exception as e:
                st.error(f"検索実行中にエラーが発生しました: {e}")

    if 'scored_chunks' in st.session_state:
        st.subheader("4. 検索結果（関連度スコア順）")
        st.write("質問と各チャンクの意味的な近さを「関連度スコア」として計算し、スコアの高い順に並べ替えました。")
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