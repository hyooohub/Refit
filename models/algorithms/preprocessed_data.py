import json
from collections import defaultdict

import os

# 1. 원시 JSON 파일 불러오기
with open("C:\\Users\\효원\\Desktop\\데이터셋\\주문.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 대화셋을 QA 페어로 정리
faq_data = []
dialogue_map = defaultdict(list)

# 대화셋을 일련번호 기준으로 그룹화
for row in raw_data:
    dialogue_map[row["대화셋일련번호"]].append(row)

# 각 대화셋마다 질문-답변 페어 추출
for convo_id, turns in dialogue_map.items():
    category = turns[0].get("카테고리", "기타")

    for i in range(len(turns) - 1):
        q_turn = turns[i]
        a_turn = turns[i + 1]

        if q_turn["화자"] == "고객" and q_turn["고객질문(요청)"].strip() \
           and a_turn["화자"] == "상담사" and a_turn["상담사답변"].strip():
            faq_data.append({
                "카테고리": category,
                "질문": q_turn["고객질문(요청)"].strip(),
                "답변": a_turn["상담사답변"].strip()
            })

# 3. 저장 - JSON
with open("C:\\Users\\효원\\Desktop\\데이터셋\\data/order_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=2)


print(f"총 {len(faq_data)}개의 QA 페어 추출 완료!")
