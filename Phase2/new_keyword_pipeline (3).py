import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
import re
from openai import AzureOpenAI
import os

# Azure OpenAI 환경 설정
os.environ['OPENAI_API_VERSION'] = '2024-02-15-preview'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://upappliance.openai.azure.com'
os.environ['AZURE_OPENAI_API_KEY'] = '4gWgDnxVjVaIRQhPJFn2seBhFGhsgfZhpedg6J0lmWDdsltOAlD8JQQJ99AKACYeBjFXJ3w3AAABACOGylYc'
os.environ['AZURE_OPENAI_API_DEPLOYMENT_NAME'] = "gpt-4o-mini"

# ChromaDB 설정
os.environ["CHROMA_TELEMETRY"] = "False"
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb

def setup_vectordb(csv_path: str) -> Chroma:
    """
    CSV 경로를 받아 벡터 DB (ChromaDB) 구축 후 반환
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    documents = [
        Document(
            page_content=row["Keyword"],
            metadata={"No": int(row["No"])}
        )
        for _, row in df.iterrows()
    ]

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        openai_api_version="2024-02-15-preview"
    )
    vectordb = Chroma.from_documents(documents, embeddings)

    print(f"✅ 벡터 DB 구축 완료! 저장된 문서 수: {len(documents)}")
    return vectordb

def format_top10_for_keyword_prompt(top10_records: list) -> str:
    """
    top10_records(list of dict) -> 프롬프트에 넣기 좋은 간단 표 문자열
    각 아이템 예시: {
        "순위": 1, "ticket_id": "...", "Cosine 유사도": 0.9342,
        "문서 요약": "...", "component_group": "..."
    }
    """
    lines = []
    for r in top10_records:
        rank = r.get("순위", "")
        tid  = r.get("ticket_id", "")
        sim  = r.get("Cosine 유사도", "")
        summ = r.get("문서 요약", "")
        # cg   = r.get("component_group", "")
        lines.append(f"- #{rank} | {tid} | sim={sim} | {summ}")
    return "\n".join(lines) if lines else "(no similar docs)"


def generate_new_keyword(
    client: AzureOpenAI,
    ticket_id: str,
    component_group: str,
    components: str,
    beforechange: str,
    afterchange: str,
    gen_sum: str,
    sim_score: float,
    sim_keyword: str,
    top10_records: list = None,       # ← 추가
    top10_table_md: str = None        # ← 선택
) -> dict:
    """
    벡터DB를 사용하지 않고, 이미 계산된 Top-10 유사 문서 목록을 기반으로
    gen_sum과 의미가 겹치지 않는 '한국어 단일 키워드'를 생성.
    """
    if top10_records is None:
        top10_records = []

    top10_block = format_top10_for_keyword_prompt(top10_records)
    if (not top10_records) and top10_table_md:
        # 마크다운만 있을 때는 그대로 붙여준다 (정규식 파싱 가능하지만 필요시 추가)
        top10_block = top10_table_md

    prompt = f"""
[System]
당신은 제품 제안(gen_sum)과 유사 문서 Top-10 요약을 비교하여,
겹치지 않는 **한국어 단일 키워드**를 하나만 생성하는 전문가입니다.

[Input]
- ticket_id: {ticket_id}
- component_group: {component_group}
- components: {components}
- gen_sum: {gen_sum}
- 기존 유사 키워드(sim_keyword): {sim_keyword}
- RAG 유사도 점수(sim_score): {sim_score}

- Top-10 유사 문서 요약(가장 유사한 순):
{top10_block}

[지침]
1) gen_sum의 핵심 의도를 반영하되, 아래 항목과 **의미적으로 중복되거나 거의 동일한 표현**을 피하세요.
   - 기존 sim_keyword
   - Top-10 문서 요약들에서 반복적으로 등장하는 기능/표현
2) 결과는 **한국어 하나의 단일 키워드**로만 출력하세요.
   - 공백/하이픈/언더스코어 없이 붙여 쓴 단어형이 이상적입니다. (예: "얼굴인식잠금", "세탁량자동감지")
   - 너무 일반적인 상위어(예: "기능개선", "편의성향상")는 피하고, **구체적이고 대표성 있는** 명사를 선택하세요.
3) 제품 맥락(가전/홈어플라이언스)과 component_group, components와도 **자연스럽게 어울리는 표현**을 쓰세요.

[형식]
아래 형식을 그대로 사용해 출력하세요. 다른 문구를 추가하지 마세요.

ticket_id: {ticket_id}
new_keyword: (한국어 단일 키워드 1개)
"""

    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    output = response.choices[0].message.content.strip()
    print(f"\n✅ GPT Output:\n{output}")

    match = re.search(r"new_keyword\s*[:：]\s*(.+)", output, re.IGNORECASE)
    if match:
        keyword = match.group(1).strip()
        keyword = keyword.replace("_", " ")
        keyword = re.sub(r"(기능|추가|개선|솔루션)$", "", keyword).strip()
        new_keyword = keyword
    else:
        new_keyword = "키워드오류"

    result = {
        "ticket_id": ticket_id,
        "component_group": component_group,
        "components": components,
        "beforechange": beforechange,
        "afterchange": afterchange,
        "gen_sum": gen_sum,
        "sim_score": sim_score,
        "sim_keyword": sim_keyword,
        "new_keyword": new_keyword,
        "top10_used": bool(top10_records) or bool(top10_table_md),

    }

    print("\n✅ [함수 내부 최종 결과]")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result

# # 실행 예시
# if __name__ == "__main__":
#     import os
#     from openai import OpenAI

#     # ✅ 환경 변수
#     os.environ["OPENAI_API_KEY"] = "sk-"

#     # ✅ 클라이언트
#     client = OpenAI()

#     # ✅ 벡터 DB 구축
#     vectordb = setup_vectordb("./keyword_list.csv")

#     # ✅ gen_sum 예시
#     gen_sum = "The user suggests adding a feature that allows the ice maker to produce ice again when the ice tray is perceived as full but has consumed some ice."

#     # ✅ Top-10 키워드 검색
#     top10_keywords = search_top10_keywords(vectordb, gen_sum, top_k=10)

#     # ✅ 기타 파라미터
#     ticket_id = "64babec"
#     components = "냉장고"
#     beforechange = "이전 변경 내용"
#     afterchange = "이후 변경 내용"
#     RAGAS_score = 95
#     sim_keyword = "아이스메이커기능개선"

#     # ✅ GPT로 새로운 키워드 생성
#     result = generate_new_keyword(
#         client,
#         ticket_id,
#         components,
#         beforechange,
#         afterchange,
#         gen_sum,
#         RAGAS_score,
#         sim_keyword,
#         top10_keywords
#         # new_keyword_score=90
#     )

