# agent.py
import google.generativeai as genai

def perspective_generation_agent(model: genai.GenerativeModel, query: str) -> list[str]:
    """ユーザーの質問から、回答すべき観点を動的に生成するエージェント"""
    prompt = f"""
質問: {query}

上記の質問に対し、回答に含めるべき**観点**を3〜5つ、箇条書きで提案してください。観点のみを列挙し、説明は不要です。

例:
- 定義
- メリット
- デメリット
- 具体例
- 注意点
"""
    response = model.generate_content(prompt)
    perspectives_raw = response.text.strip()
    perspectives = [line.lstrip('- ').strip() for line in perspectives_raw.split('\n') if line.strip()]
    return perspectives


def generic_perspective_agent(model: genai.GenerativeModel, query: str, perspective: str) -> str:
    """特定の観点に基づいて回答を生成する汎用エージェント"""
    prompt = f"""以下の質問と観点に基づいて、具体的な情報を提供してください。

    質問: {query}
    観点: {perspective}

    この観点に特化して、簡潔かつ分かりやすく回答してください。
    """
    response = model.generate_content(prompt)
    return response.text

def overall_agent(model: genai.GenerativeModel, original_query: str, perspective_responses: dict[str, str]) -> str:
    """各エージェントの回答を統合し、最終的な回答を生成する統括エージェント"""
    agent_responses_str = "\n\n".join([f"**{p}に関する情報:**\n{r}" for p, r in perspective_responses.items()])

    prompt = f"""ユーザーからの元の質問:
    「{original_query}」

    以下の各エージェントが提供した情報を総合的に考慮し、元の質問に対する最終的で詳細な回答を作成してください。
    各観点からの情報を統合し、論理的で分かりやすい文章でまとめてください。

    --- 各エージェントからの情報 ---
    {agent_responses_str}
    ---

    最終的な回答:
    """
    response = model.generate_content(prompt)
    return response.text