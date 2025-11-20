本アプリではRAGの基本的な処理の流れを学ぶことができます。
RAG_Tutorial.ipynbで一つ一つの処理を解説し、RAG_Tutorial.pyでそれらをもとに一つのRAGアプリを作成しています。
課題としては、RAG_Tutorial.homework.py中にコードを削除した部分が5つあるので、RAG_Tutorial.ipynbやStreamlitの公式ドキュメントを参考に削除された部分を補完してください。なおRAG_Tutorial.pyが回答の役割を果たします。

追加の発展要素としてsupabaseをベクトルDBとして使用する方法やPDFドキュメントを読み込み、テキスト情報が無いものはGoogle Cloud Vision APIでOCR処理をする方法を追加したアプリとしてRAG_APP_fullmodel.pyを用意しています。最終発表のアプリ作成や自主学習に役立ててください。
また、RAGの精度を向上させたい場合は、以下のような手法があります。興味のある方はぜひ挑戦してみてください。
難易度,手法,概要
低 🔰,チャンク戦略の最適化,チャンクサイズとオーバーラップを調整する
低 🔰,埋め込みモデルの変更,より高性能なEmbeddingモデルに入れ替える
中 🧑‍💻,Re-ranking (リランキング),取得したチャンクを、再度並べ替えて絞り込む
中 🧑‍💻,Hybrid Search (ハイブリッド検索),ベクトル検索とキーワード検索（BM25等）を組み合わせる
中 🧑‍💻,Query Expansion (クエリ拡張),ユーザーの質問をLLMで複数パターンに書き換えて検索する
高 🚀,Parent Document Retriever,検索は小さなチャンクで、LLMにはその親（大きな）チャンクを渡す
高 🚀,Sentence-Window Retrieval,検索は「文」単位で、LLMにはその「前後」を含めて渡す
