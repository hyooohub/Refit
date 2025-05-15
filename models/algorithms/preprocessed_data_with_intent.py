import json
from collections import defaultdict

# 1. 원시 JSON 파일 불러오기
with open("C:\\Users\\효원\\Desktop\\데이터셋\\raw_data\\결제.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 2. 대화셋을 QA 페어로 정리
faq_data = []
dialogue_map = defaultdict(list)

# 대화셋을 일련번호 기준으로 그룹화
for row in raw_data:
    dialogue_map[row["대화셋일련번호"]].append(row)

# 각 대화셋마다 질문-답변 페어 추출
for convo_id, turns in dialogue_map.items():
    for i in range(len(turns) - 1):
        q_turn = turns[i]
        a_turn = turns[i + 1]

        question = q_turn.get("고객질문(요청)", "").strip()
        answer = a_turn.get("상담사답변", "").strip()
        intent = q_turn.get("고객의도", "").strip()

        if (
            q_turn.get("화자") == "고객"
            and a_turn.get("화자") == "상담사"
            and question
            and intent
            and answer
            and len(answer) >= 20  # 답변이 10자 이상인 경우만
        ):
            faq_data.append({
                "질문": question,
                "답변": answer,
                "고객의도": intent
            })

# 3. 저장 - JSON
save_path = "C:\\Users\\효원\\Desktop\\데이터셋\\data\\payment_cleaned_with_intent.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(faq_data, f, ensure_ascii=False, indent=2)

print(f"✅ 고객의도 포함 + 답변 10자 이상 {len(faq_data)}개의 QA 페어 저장 완료!")
