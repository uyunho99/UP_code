# %%
import os
# Azure OpenAI 환경 설정
os.environ['OPENAI_API_VERSION'] = '2024-02-15-preview'
os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://upappliance.openai.azure.com'
os.environ['AZURE_OPENAI_API_KEY'] = '4gWgDnxVjVaIRQhPJFn2seBhFGhsgfZhpedg6J0lmWDdsltOAlD8JQQJ99AKACYeBjFXJ3w3AAABACOGylYc'
os.environ['AZURE_OPENAI_API_DEPLOYMENT_NAME'] ="gpt-4o-mini"

# -*- coding: utf-8 -*-
# =========================================================
# ICC / RAG / Self-Consistency (Top-2) Pipeline
# - 요청사항: sc_prompt / choose_prompt 결과를 "파싱 없이" 원문 그대로 저장
# - Role/Task 문구는 유지, 출력 포맷만 JSON(Top-2)으로 유도하되 코드에서는 파싱하지 않음
# =========================================================

# In[1]: 기본 환경 설정 --------------------------------------------------------
import os, sys, re, json, time, math
import pandas as pd

# (권장) 환경변수는 OS 환경에 미리 세팅하세요. 여기서는 예시만 남깁니다.
# os.environ['OPENAI_API_VERSION'] = '2024-02-15-preview'
# os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://<your-endpoint>.openai.azure.com'
# os.environ['AZURE_OPENAI_API_KEY'] = '<your-key>'
# os.environ['AZURE_OPENAI_API_DEPLOYMENT_NAME'] ="gpt-4o-mini"

# 선택: ChromaDB 사용 시 텔레메트리 비활성화
os.environ["CHROMA_TELEMETRY"] = "False"

import pysqlite3  # chroma 환경에서 sqlite 대체 사용 시
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings

from langchain_community.chat_models import ChatOpenAI as LegacyChatOpenAI  # (미사용)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnableMap, RunnableLambda
from langchain.callbacks import get_openai_callback
from langchain.schema import Document

# In[2]: 타이밍 데코레이터 ------------------------------------------------------
timing_results = []
def timed(name):
    def wrapper(fn):
        def inner(x):
            print(f"⏱️ [{name}] 시작")
            start = time.time()
            result = fn(x)
            end = time.time()
            duration = end - start
            print(f"✅ [{name}] 완료 - 소요 시간: {duration:.2f}초")
            global timing_results
            timing_results.append((name, duration))
            return result
        return inner
    return wrapper

# In[3]: 데이터 로드 -----------------------------------------------------------
# df = pd.read_csv('../../../flow/1_개발폴더_한양대/Phase2_RAG/train_set.csv', encoding='utf-8-sig')
df = pd.read_csv('/datasets/DTS2025000169/data/trainset_group.csv', encoding='utf-8-sig')

# ICC 결과 저장용(옵션)
icc_df = pd.DataFrame(columns=["ticket_id", "components", "before_change", "after_change", "ICC"])

# In[4]: LLM 인스턴스 ----------------------------------------------------------
# LangChain의 AzureChatOpenAI 사용 (chat.completions 호환)
llm = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_API_DEPLOYMENT_NAME", "gpt-4o-mini"),
    openai_api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# In[5]: 임베딩 & ChromaDB 컬렉션 로드 ------------------------------------------
aoai_embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    openai_api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview"),
)

# (초기 구축은 별도 함수로 실행 후, 여기서는 PersistentClient로 로드)
client = PersistentClient(path="./chromadb_store_1")
native_collection = client.get_collection("my_native_collection")

# ticket_id -> keyword 매핑 (있으면 사용)
try:
    TICKET_TO_KEYWORD = dict(zip(df["ticket_id_hashed"], df.get("keyword", pd.Series(["N/A"] * len(df)))))
except Exception:
    TICKET_TO_KEYWORD = {}

# In[6]: 프롬프트 정의 ----------------------------------------------------------

# (1) ICC 분류: 예시/판별기준 강화. 출력은 "Suggestion" 또는 "ICC" 한 단어
icc_prompt = PromptTemplate.from_template("""
<Role> You are an expert in reading and classifying product feedback.</Role>

<Task>
Decide whether the user feedback is a "Suggestion" (new feature request / improvement) or "ICC".
- If the feedback includes **new feature requests or improvement suggestions**, classify as "Suggestion".
- Otherwise, if it fits Issues/Complaints/Comments (ICC), classify as "ICC".
</Task>

<Definitions with Examples>
[Issues (오류/고장/결함/에러)]
- Signals: "오류", "고장", "에러", "업데이트 안 됨", "연결 안 됨", "삭제됨", unexpected on/off, error codes, etc.
- Examples:
  - "앱에서 필터를 재설정하려고 하면 '네트워크를 찾을 수 없음' 오류가 납니다."
  - "건조기가 시작되지 않고 E31 오류 코드가 표시됩니다."
  - "워시타워 건조기 업그레이드 완료했는데도 업가전센터에서는 계속 업그레이드로 뜹니다."
  - "업데이트할 때마다 연결이 안 됩니다. 프로그램 실력이 없으면 그대로 사용하게 해주세요."
  - "오븐 불이 저절로 켜집니다."

[Complaints (불만/불편/실망/짜증)]
- Signals: 짜증/불편/실망/어처구니없음 등 감정 중심의 부정적 표현
- Examples:
  - "쓸데없는 알림 보내지 마세요. 다시는 이 브랜드 안 씁니다."
  - "시작/종료 버튼을 직접 눌러야 한다니 IoT가 어처구니없네요."
  - "음성인식 좀 개선하세요. G사 스피커는 아이 말도 잘 인식돼요."
  - "이런 제품에 2000달러를 쓰게 하나요?"
  - "건조 성능이 형편없습니다."

[Comments (문의/중립적 건의/질문)]
- Signals: 감정/결함 없이 정보/질문/설명 중심
- Examples:
  - "관심 없는 업데이트는 숨길 수 있게 해주세요. 세탁기 알림음 다양하게 필요 없어요."
  - "사전세탁 사이클을 어떻게 켜나요? 옵션이 꺼져 있고 켜짐이 안 됩니다."
  - "로봇 청소시간에 홈뷰/리모컨으로 청소구역 설정할 수 있나요?"
  - "내 세탁기에 대한 업데이트를 받아볼 수 있을까요?"

<Output>
Return ONLY one word:
- "Suggestion" OR "ICC"
No extra text.

Input Summary:
{original_input}
""")

# (2) 통합 제안 분리 및 요약 (영어) — JSON 배열로 유도 (코드는 파싱하지 않아도 OK)
combined_suggestion_prompt = PromptTemplate.from_template("""
<Role> You are an expert in understanding and extracting user suggestions from product feedback, specializing in home appliances. </Role>

<Task> Your task is to perform the following steps:
1.  Carefully read the provided user feedback, which might contain multiple distinct suggestions.
2.  **Separate each distinct suggestion clearly.** Identify and number each unique suggestion.
3.  For each identified suggestion, summarize it comprehensively while maintaining its distinctiveness and context, especially with respect to the component `{components}`.
4.  **Ensure all summarized suggestions are output in English.** Do not merge different suggestions into one if they are distinct.
5. The summarized sentences would be started by 'The user suggested'.

Reasoning: First, identify if the input contains more than one distinct idea (look for keywords like "and", punctuation, or multiple sentences indicating separate ideas). Then, for each distinct idea, summarize it comprehensively in English, ensuring it stands as a standalone improvement point related to `{components}`.

Text:
{original_input}

<Format>: Strictly follow these output rules:
- Output must be a valid JSON array of strings.
- The output must start with `[` and end with `]`, and contain no other text before or after.
- Do not include any explanations, labels, metadata, or extra text.
- Each string in the array should be a distinct, summarized suggestion in **English**.
</Format>
""")

# (3) 첫 번째 제안 선택 (원문 유지)
first_prompt = PromptTemplate.from_template("""<Role> You select the first suggestion from the comprehensive summary based on order of appearance. </Role>
<Task>: Identify and return the first suggestion from the comprehensive summary, exactly as it appears, without any additional text.

Reasoning: Use a Thought step to parse the comprehensive summary and find the first suggestion mentioned. Then Act by outputting that first suggestion verbatim. Do not include any formatting, numbering, or other suggestions.

Comprehensive Summary: 
{overall_summary}  

List of Original Suggestions:
{proposals}

<Format>: Output the first Suggestion as a plain text string (no JSON, no list formatting, no quotes around it).""")

sc_prompt = PromptTemplate.from_template("""
<Role> You are an expert who independently evaluates the most similar document based on the Suggestion statement and top 10 similarity documents. This process will be repeated 3 times to ensure Self-Consistency. </Role>

<Task>: Based on the following information, select the most similar document. Evaluate each criterion below with numeric scores (integers), and calculate the total evaluation score and thinking confidence at the end.

[Input Information]
Suggestion:
{proposal_summary}

Top 10 Documents:
{top_10_table}

<Evaluation Criteria>: Evaluate each criterion with quantitative scores.

- Selected Document (Ticket_id): [e.g., fdff64d]
- Functional Category Alignment (Context Entity Recall): [0-25 points]
- Claim Coverage (Context Recall): [0-25 points]
- Evidence Faithfulness: [0-25 points]
- Explanation Flow Similarity: [0-25 points]

<Qualitative Assessment Criteria>: Self-evaluate your reasoning process.

- Clarity of Criteria Application: [0-30 points]
- Logical Coherence: [0-30 points]
- Persuasiveness of Explanation: [0-20 points]
- Absence of Ambiguity: [0-20 points]

<Output JSON Schema>
Return ONLY a valid JSON object in the following format:
{{
  "top2": [
    {{
      "selected_ticket_id": "string",
      "scores": {{
        "func_align": int,
        "claim_coverage": int,
        "faithfulness": int,
        "flow_similarity": int,
        "total": int
      }},
      "self_eval": {{
        "clarity": int,
        "logic": int,
        "persuasion": int,
        "no_ambiguity": int,
        "confidence_pct": int
      }},
      "reason": "string"
    }},
    {{
      "selected_ticket_id": "string",
      "scores": {{
        "func_align": int,
        "claim_coverage": int,
        "faithfulness": int,
        "flow_similarity": int,
        "total": int
      }},
      "self_eval": {{
        "clarity": int,
        "logic": int,
        "persuasion": int,
        "no_ambiguity": int,
        "confidence_pct": int
      }},
      "reason": "string"
    }}
  ]
}}

<Rules>
- Output ONLY the JSON object (no markdown, no extra text).
- The two "selected_ticket_id" MUST be from the provided Top 10.
- Order the two items by your preference (best first).
""")

choose_prompt = PromptTemplate.from_template("""
<Role7> You are an expert who integrates three repeated evaluation results and determines the most reliable final recommended document. </Role7>

<Task>: Analyze the three Self-Consistency-based evaluation responses together with RAG-based similarity scores and RAGAS-based quantitative metrics to ultimately select the most similar document and explain your reasoning.

Reasoning: Collect all the following metrics to calculate an integrated score, then select the most comprehensively similar document.

<Integrated Evaluation Criteria>
1. Self-Consistency Average Confidence (Weight 0.3)
2. Same Document Repeated Selection (+10 point bonus)
3. Self-Consistency Average Total Score (Weight 0.2)
4. RAG-based Cosine Similarity (Weight 0.2)
5. Faithfulness (RAGAS metric) (Weight 0.15)
6. Context Recall (RAGAS metric) (Weight 0.15)

<Final Selection Criteria>
- Select the document with the highest integrated score.
- In case of equal scores, prioritize documents more frequently selected in Self-Consistency.
- The decision should be based on how similar the document summary, functional purpose, and explanation style are to the Suggestion statement.

Self-Consistency Response List:
{self_consistency_responses}

RAG Similarity List:
{top_10_table}

<Output JSON Schema>
Return ONLY a valid JSON object like:
{{
  "final_ticket_id": "string",
  "final_top2": [
    {{
      "ticket_id": "string",
      "rag_similarity_pct": number,
      "integrated_score": number,
      "reasons": [
        "string"
      ]
    }},
    {{
      "ticket_id": "string",
      "rag_similarity_pct": number,
      "integrated_score": number,
      "reasons": [
        "string"
      ]
    }}
  ],
  "notes": "string"
}}

<Rules>
- You MUST output exactly two items in "final_top2", ordered best-first.
- Strongly consider frequency across the three Self-Consistency runs (e.g., a ticket repeatedly selected should outrank others), then apply the integrated score.
- "final_ticket_id" must equal the first element's "ticket_id".
- Derive rag_similarity_pct from the given RAG table for each chosen ticket (0~1 → ×100).
- Output ONLY the JSON object (no extra text).
""")



# In[7]: 헬퍼 함수들 -------------------------------------------------------------
def retrieve_context_native(query: str, component_group: str, collection, embed_model, ticket_to_keyword: dict = None):
    """ChromaDB 네이티브 컬렉션에서 유사 문서를 검색하고,
       component_group/keyword/유사도를 포함한 테이블 & 레코드 & 문서 리스트를 반환."""
    if ticket_to_keyword is None:
        ticket_to_keyword = TICKET_TO_KEYWORD

    # 1) 쿼리 임베딩
    if hasattr(embed_model, "embed_query"):
        query_embedding = embed_model.embed_query(query)
    else:
        query_embedding = embed_model.encode(query).tolist()

    # 2) where 필터
    where_filter = {"component_group": component_group} if component_group else {}

    # 3) 컬렉션 질의
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where=where_filter
    )

    if not results["ids"] or not results["ids"][0]:
        return "검색 결과가 없습니다.", [], [], "[]"

    rows, docs, records = [], [], []
    for idx, (doc_id, dist, meta, content) in enumerate(zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
        results["documents"][0]
    ), start=1):
        ticket_id = meta.get("ticket_id", "N/A")
        doc_cg = meta.get("component_group", "N/A")
        kw = ticket_to_keyword.get(ticket_id, "N/A")
        summary = str(content).strip().replace("\n", " ")[:120]
        similarity = 1 - float(dist)  # cosine distance → similarity

        rows.append(f"| {idx} | {ticket_id} | {doc_cg} | {kw} | {similarity:.4f} | {summary} |")
        records.append({
            "rank": idx,
            "ticket_id": ticket_id,
            "component_group": doc_cg,
            "keyword": kw,
            "similarity": round(similarity, 4),  # 0~1
            "summary": summary
        })
        docs.append(Document(page_content=content, metadata=meta))

    table = "\n".join([
        "| 순위 | Ticket ID | Component Group | Keyword | Cosine 유사도 | 문서 요약 |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows
    ])
    top10_json = json.dumps(records, ensure_ascii=False)
    return table, docs, records, top10_json


def update_icc_df(x):
    """ICC로 판정된 경우 icc_df에 추가 저장 (제안(Suggestion)이면 미수행)"""
    if x.get("icc_check") != "ICC":
        return x  # Suggestion이면 아무 것도 안 하고 반환
    new_row = {
        "ticket_id": x.get("ticket_id_hashed", ""),
        "components": x.get("components", ""),
        "before_change": x.get("beforechange", ""),
        "after_change": x.get("afterchange", ""),
        "ICC": "ICC"
    }
    global icc_df
    icc_df = pd.concat([icc_df, pd.DataFrame([new_row])], ignore_index=True)
    return x


token_usage_results = []
def invoke_llm_with_token_tracking(prompt_template, input_data, step_name):
    """LLM 호출 + 토큰 사용량 기록 — 응답은 '원문 그대로 문자열'로 반환"""
    with get_openai_callback() as cb:
        resp = llm.invoke(prompt_template.format(**input_data))
    token_usage_results.append({
        "step": step_name,
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
    })
    # LangChain 메시지 객체면 .content, 문자열이면 그대로 반환
    return getattr(resp, "content", resp)


def select_rag_top2(records):
    """
    records: retrieve_context_native()가 반환한 dict 리스트
             [{ticket_id, component_group, keyword, similarity(0~1), summary}, ...]
    return:
      final_result_json (str), top1_ticket_id (str), top1_keyword (str), top1_score_pct (float)
    """
    if not records:
        final_obj = {"final_ticket_id": "", "final_top2": [], "notes": "No RAG results"}
        return json.dumps(final_obj, ensure_ascii=False), "", "", 0.0

    sorted_rec = sorted(records, key=lambda r: r.get("similarity", 0), reverse=True)[:2]
    top2 = []
    for r in sorted_rec:
        pct = round(float(r.get("similarity", 0)) * 100, 2)
        top2.append({
            "ticket_id": r.get("ticket_id", ""),
            "rag_similarity_pct": pct,
            # Choose 단계를 생략하므로 integrated_score는 간단히 rag%로 둠
            "integrated_score": pct,
            "reasons": ["Selected solely by RAG cosine similarity (highest first)."]
        })

    final_obj = {
        "final_ticket_id": top2[0]["ticket_id"] if top2 else "",
        "final_top2": top2,
        "notes": "Self-Consistency/Choose prompts skipped; RAG-only selection."
    }
    top1_id = sorted_rec[0].get("ticket_id", "") if sorted_rec else ""
    top1_kw = sorted_rec[0].get("keyword", "") if sorted_rec else ""
    top1_pct = round(float(sorted_rec[0].get("similarity", 0)) * 100, 2) if sorted_rec else 0.0
    return json.dumps(final_obj, ensure_ascii=False), top1_id, top1_kw, top1_pct


# In[8]: 체인 정의 -------------------------------------------------------------
def _icc_branch(x):
    if x["icc_check"] == "ICC":
        y = update_icc_df(x)
        print("👉 판별 결과 : ICC")
        y = y | {"__is_icc": True}
        return y
    else:
        print("👉 판별 결과 : Proposal")
        return x | {"__is_icc": False}

chain = RunnableSequence(
    # Step 0: 입력받기
    RunnableLambda(timed("입력받기")(lambda x: x)),

    # Step 1: ICC 분류
    RunnableLambda(timed("ICC 분류")(lambda x: x | {
        "icc_check": invoke_llm_with_token_tracking(
            prompt_template=icc_prompt,
            input_data={"original_input": x["original_input"]},
            step_name="ICC 분류"
        ).strip(),
        "component_group": x["component_group"]
    })),

    # Step 2: ICC 판단 및 분기 (플래그만 세움)
    RunnableLambda(timed("ICC 분기처리")(_icc_branch)),

    # Step 3: 제안 분리 및 전체 제안 요약 (영어)
    RunnableLambda(timed("통합 제안 분리 및 요약 (영어)")(lambda x: x | {
        "proposal_summary_all": invoke_llm_with_token_tracking(
            prompt_template=combined_suggestion_prompt,
            input_data={"original_input": x["original_input"], "components": x["components"]},
            step_name="통합 제안 분리 및 요약 (영어)"
        ),
        "component_group": x["component_group"]
    })),

    # Step 4: 첫 번째 제안 선택
    RunnableLambda(timed("첫번째 제안 선택")(lambda x: x | {
        "first_proposal": invoke_llm_with_token_tracking(
            prompt_template=first_prompt,
            input_data={
                "overall_summary": x["proposal_summary_all"],   # 파싱 없이 그대로 전달
                "proposals": x["proposal_summary_all"]          # 파싱 없이 그대로 전달
            },
            step_name="첫번째 제안 선택"
        ).strip(),
        "component_group": x["component_group"]
    })),

    # Step 5: Top-10 유사 문서 검색 (component_group + keyword 포함, JSON도 생성)
    RunnableLambda(timed("Top-10 유사 문서 검색")(lambda x: x | (lambda table, docs, records, top10_json: {
        "top_10_table": table,
        "top_10_docs": docs,
        "top_10_records": records,
        "top_10_json": top10_json,
        "component_group": x["component_group"]
    })(*retrieve_context_native(
        query=x["first_proposal"],
        component_group=x["component_group"],
        collection=native_collection,
        embed_model=aoai_embeddings,
        ticket_to_keyword=TICKET_TO_KEYWORD
    )))),

    # Step 6: ✅ RAG Top-2 직접 선택 (SC/Choose 생략)
    RunnableLambda(timed("RAG Top-2 선택")(lambda x: (lambda final_json, top1_id, top1_kw, top1_pct: x | {
        "final_result": final_json,          # 직접 만든 JSON 문자열
        "ticket_id": top1_id,                # 상위 1개 ticket_id
        "sim_keyword": top1_kw,              # 상위 1개 문서의 keyword
        "sim_score": top1_pct                # 상위 1개 유사도(%) → 후속 로직의 90컷과 호환
    })(*select_rag_top2(x["top_10_records"]))))
)


# In[9]: main() ----------------------------------------------------------------
def main(inputs):
    """
    inputs 예시:
    {
        "original_input": "ThinkQ 평면도상에 선을 그어 청소구역을 지정하도록 해주세요",
        "components": "로봇청소기",
        "component_group": "robot",
        "afterchange": "",
        "beforechange": "",
        "ticket_id_hashed": "xxxx"
    }
    """
    global timing_results, token_usage_results, icc_df
    timing_results = []
    token_usage_results = []

    print("\n📥 입력 제안문:")
    print(inputs["original_input"])

    result = chain.invoke(inputs)
    print("✅ result keys:", list(result.keys()))

    # ICC인 경우: 최소 정보로 반환 (RAG/요약 스킵)
    if result.get("__is_icc"):
        print("\n🔔 ICC로 판정되어 RAG Top-2 선택을 수행하지 않습니다.")
        final_json = json.dumps(
            {"final_ticket_id": "", "final_top2": [], "notes": "ICC item"},
            ensure_ascii=False
        )
        return (
            inputs.get("ticket_id_hashed", ""),      # 1 ticket_id
            inputs.get("components", ""),            # 2 components
            inputs.get("beforechange", ""),          # 3 before_change
            inputs.get("afterchange", ""),           # 4 after_change
            "",                                      # 5 generated_summary
            0.0,                                     # 6 sim_score
            "",                                      # 7 sim_keyword
            timing_results,                          # 8 timing_results
            "",                                      # 9 proposal_summary_all
            inputs.get("component_group", ""),       # 10 component_group
            final_json,                              # 11 final_result (JSON)
            token_usage_results,                     # 12 token_usage_results
            icc_df.copy(),                           # 13 icc_df
            "",                                      # 14 top_10_table
            []                                       # 15 top_10_records
        )

    # 일반(Suggestion) 흐름: 요약/Top-10/RAG Top-2 결과 출력
    print("\n📄 전체 제안 요약(LLM 원문 그대로):")
    print(result["proposal_summary_all"])

    print("\n⭐ 첫 번째 제안(LLM 원문 그대로):")
    print(result["first_proposal"])

    print("\n🔝 Top-10 유사 문서 테이블:")
    print(result["top_10_table"])

    print("\n🏁 최종 결정(RAG Top-2 JSON):")
    print(result["final_result"])

    # --- 토큰 사용량 요약 ---
    print("\n--- 📊 토큰 사용량 요약 ---")
    total_prompt_tokens = sum(t.get("prompt_tokens", 0) for t in token_usage_results)
    total_completion_tokens = sum(t.get("completion_tokens", 0) for t in token_usage_results)
    total_tokens_overall = sum(t.get("total_tokens", 0) for t in token_usage_results)

    print(f"총 프롬프트 토큰: {total_prompt_tokens}")
    print(f"총 완성 토큰: {total_completion_tokens}")
    print(f"전체 토큰 사용량: {total_tokens_overall}")
    print("단계별 토큰 사용량:")
    for entry in token_usage_results:
        print(f"- {entry['step']}: 총 {entry['total_tokens']} (프롬프트: {entry['prompt_tokens']}, 완성: {entry['completion_tokens']})")

    # --- 반환 (정확히 15개) ---
    ticket_id = result.get("ticket_id", "")
    components = inputs.get("components", "")
    before_change = inputs.get("beforechange", "")
    after_change = inputs.get("afterchange", "")
    generated_summary = result.get("first_proposal", inputs.get("generated_summary", ""))
    sim_score = float(result.get("sim_score", 0.0))        # Top-1 유사도(%)
    sim_keyword = result.get("sim_keyword", "")            # Top-1 keyword
    proposal_summary_all = result.get("proposal_summary_all", "")
    component_group = result.get("component_group", inputs.get("component_group", ""))
    top_10_table = result.get("top_10_table", "")
    top_10_records = result.get("top_10_records", [])
    final_result = result.get("final_result", "")          # RAG Top-2 JSON 문자열

    return (
        ticket_id,                    # 1
        components,                   # 2
        before_change,                # 3
        after_change,                 # 4
        generated_summary,            # 5
        sim_score,                    # 6  (Top-1 % → 기존 90 컷과 호환)
        sim_keyword,                  # 7  (Top-1의 keyword)
        timing_results,               # 8
        proposal_summary_all,         # 9
        component_group,              # 10
        final_result,                 # 11 (RAG Top-2 JSON)
        token_usage_results,          # 12
        icc_df.copy(),                # 13
        top_10_table,                 # 14
        top_10_records                # 15
    )

if __name__ == "__main__":
    main()
