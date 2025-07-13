import streamlit as st
import google.generativeai as genai
import os
import json
from datetime import datetime

# agent.pyからエージェント関数をインポート
from agent import perspective_generation_agent, generic_perspective_agent, overall_agent

# --- 定数と初期設定 ---
HISTORY_FILE = "question_history.json" # 履歴を保存するファイル名 (JSON形式に変更)
# Gemini APIキーの設定
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Geminiモデルの初期化 (main.pyで一度だけ行う)
model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')

st.set_page_config(page_title="AIナレッジハブ", layout="centered") # タイトル変更

st.title("💡 AI ナレッジハブ") # タイトル変更
st.write("""AIナレッジハブは、どんな質問にも多角的に答える汎用AIエージェントアプリです。""")
st.write("""
質問内容に応じて、複数の専門AIが深く分析し、結果を統括AIがまとめて最適な回答を生成します。
あなたはどんな複雑な問いに対しても、質の高い回答を瞬時に得ることができます。""") # 説明文変更

# --- アプリ起動時の履歴リセット ---
if "app_initialized" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE) # 履歴ファイルを削除
    st.session_state.app_initialized = True
    st.session_state.user_query_text = "" # 質問入力欄も初期化


# --- 質問テンプレートの定義 ---
QUESTION_TEMPLATES = {
    "プログラミング学習の最適な始め方": "プログラミング学習を始めるための最適な方法や言語について教えてください。全くの初心者です。",
    "Pythonデータ分析入門ロードマップ": "Pythonを使ってデータ分析を始めるための学習ロードマップと、役立つライブラリについて教えてください。",
    "Webデザイナー向けポートフォリオ例": "Webデザイナー志望のポートフォリオサイトに含めるべき要素と、参考になるデザイン例を教えてください。",
    "オンラインストア開設とプラットフォーム": "オンラインストアを立ち上げるための具体的な手順と、おすすめのEコマースプラットフォームを教えてください。販売する商品は手作りのアクセサリーです。",
    "効率的な部屋の片付けステップ": "散らかった部屋を効率的に片付けるためのステップと、きれいを保つコツを教えてください。",
    "日常生活で実践できる節約術": "日常生活で無理なくできる節約術をいくつか教えてください。食費と光熱費を抑えたいです。",
    "一人暮らしの防災グッズと使い方": "一人暮らしの防災対策として、最低限揃えておくべき防災グッズと、その使い方を教えてください。",
    "安眠に繋がる習慣と睡眠の質改善": "夜なかなか寝付けないので、安眠に繋がる習慣や、寝る前に避けるべきことを教えてください。",
    "効果的な英語リスニング学習法": "英語のリスニング力を向上させたいです。効果的な学習方法や、おすすめのアプリ、教材があれば教えてください。",
    "週末を充実させる気分転換アイデア": "家で過ごす週末で、気分転換になるようなクリエイティブな趣味や、リラックスできる過ごし方を教えてください。",
    "不用品を賢くお得に処分する方法": "使わなくなった家具や家電を、お得に、または環境に優しく処分する方法について教えてください。",
    "スマホを減らすデジタルデトックス": "スマートフォンの使用時間を減らして、デジタルデトックスを始めるための具体的な方法を教えてください。",
}

# --- 履歴管理関数 ---
def load_history() -> list[dict]:
    """履歴ファイルから質問と回答のペアを読み込む"""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return history

def save_history(query: str, answer: str):
    """新しい質問と回答を履歴ファイルに追記する"""
    history = load_history() # 現在の履歴をロード
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # 現在のタイムスタンプを取得
    new_entry = {"timestamp": timestamp, "question": query, "answer": answer}
    history.append(new_entry) # 新しいエントリを追加
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4) # 整形して保存


# --- 質問テンプレートボタンをフォームの外に配置 ---
st.subheader("🚀 テンプレート") # 見出し変更
st.write("""テンプレートの中からあなたの知りたいことに近いものを選んで、質問してみてください。""")
# 1行に配置するボタンの数
BUTTONS_PER_ROW = 3

# QUESTION_TEMPLATESの項目をチャンクに分割
template_items = list(QUESTION_TEMPLATES.items())
for i in range(0, len(template_items), BUTTONS_PER_ROW):
    # 各行のボタンのチャンクを取得
    row_items = template_items[i : i + BUTTONS_PER_ROW]
    
    # st.columnsで列を作成し、各ボタンを配置
    cols = st.columns(len(row_items))
    for j, (template_name, template_text) in enumerate(row_items):
        with cols[j]:
            if st.button(template_name, key=f"template_button_{i+j}"):
                st.session_state.user_query_text = template_text
                st.rerun()


# セッションステートの初期化（初回のみ）
if "user_query_text" not in st.session_state:
    st.session_state.user_query_text = ""

# --- 過去の質問履歴をサイドバーにロードしてボタンで表示 ---
with st.sidebar:
    st.header("⏳ 過去の質問履歴") # ヘッダー変更
    st.write("""ここには過去の質問が表示されます。""")
    st.write("""※ 履歴情報はアプリを起動時にリセットされます。 """)
    history_data = load_history()
    

    if history_data:
        for i, item in enumerate(history_data):
            display_string = f"{item.get('timestamp', 'タイムスタンプなし')} - {item.get('question', '質問なし')}"
            if st.button(display_string, key=f"history_button_{i}"):
                st.session_state.user_query_text = item.get('question', '')
                st.rerun()
    else:
        st.info("質問履歴がありません。") # 表示方法変更


# フォームの作成
with st.form("general_query_form"):
    st.subheader("📝 質問入力") # 見出し変更

    user_query = st.text_area(
        "テンプレートを選択するか、質問内容を入力してください", # ラベル変更
        value=st.session_state.user_query_text,
        placeholder="例: Webサイト作成の基本的なステップについて教えてください。ターゲットは初心者で、個人利用目的です。",
        height=150,
        key="main_query_textarea"
    )

    submitted = st.form_submit_button("回答生成") # ボタンテキスト変更

# --- エージェントの関数を呼び出し ---
if submitted:
    if not user_query:
        st.warning("質問内容を入力してください。") # メッセージ修正
    else:
        st.markdown("---") # 区切り線を追加し、全体をスッキリ見せる

        try:
            # 観点生成フェーズ
            with st.status("🔍 質問の観点を分析中...", expanded=True) as status: # st.statusを使用
                st.write("AIが最適な分析観点を特定しています...")
                perspectives = perspective_generation_agent(model, user_query) # modelを引数として渡す
                if perspectives:
                    st.write(f"**特定された観点:** {', '.join(perspectives)}")
                    status.update(label="✅ 観点分析が完了しました。", state="complete", expanded=False)
                else:
                    st.error("質問から有効な観点を生成できませんでした。質問を具体的にしてください。")
                    status.update(label="⚠️ 観点分析失敗", state="error", expanded=False)
                    st.stop() # 観点がない場合はここで処理を停止

            st.markdown("---")
            st.subheader("✨ 各エージェントからの回答") # 見出し変更

            perspective_responses = {}
            for i, perspective in enumerate(perspectives):
                with st.status(f"🤖 エージェントが「{perspective}」について回答を生成中...", expanded=False) as status: # st.statusを使用
                    response_text = generic_perspective_agent(model, user_query, perspective) # modelを引数として渡す
                    perspective_responses[perspective] = response_text
                    st.write(response_text) # st.writeをstatusブロック内に移動
                    status.update(label=f"✅ 「{perspective}」回答が完了しました。", state="complete", expanded=False)

            st.markdown("---")
            st.success("🎉 全ての情報収集が完了しました。統括AIが最終回答を生成します！") # メッセージ修正

            # 統括エージェントフェーズ
            with st.status("🧠 統括AIが最終回答を生成中...", expanded=True) as status: # st.statusを使用
                st.write("複数のエージェントからの情報を統合し、詳細な回答を作成しています...")
                final_answer = overall_agent(model, user_query, perspective_responses) # modelを引数として渡す
                st.markdown("---")
                st.subheader("💡 最終回答") # 見出し変更
                st.write(final_answer)
                status.update(label="✅ 最終回答が生成完了しました。", state="complete", expanded=False)

                # --- 最終回答が生成された後に履歴を保存 ---
                save_history(user_query, final_answer)
                st.toast("履歴に保存しました！", icon="💾") # よりユーザーフレンドリーな通知に変更

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.info("APIキーが正しく設定されているか、インターネット接続を確認してください。またはモデルへのリクエストが多すぎる可能性があります。")