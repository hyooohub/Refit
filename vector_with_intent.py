from sentence_transformers import SentenceTransformer
import json

# 모델 로드
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 예시 데이터셋 로드
with open("C:\\Users\\효원\\Desktop\\데이터셋\\data\\payment_cleaned_with_intent.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# 임베딩된 데이터를 저장할 리스트
processed_data = []

# 질문 및 고객의도 임베딩
for item in cleaned_data:
    question = item["질문"]
    intent = item.get("고객의도", "").strip() or "기타"  # 비어있으면 "기타"로 대체

    # 임베딩
    question_embedding = model.encode(question).tolist()
    intent_embedding = model.encode(intent).tolist()

    processed_data.append({
        "질문": question,
        "embedding": question_embedding,
        "고객의도": intent,
        "intent_embedding": intent_embedding
    })

# 저장할 JSON 구조
output_data = {
    "questions": processed_data
}

# 결과 저장
with open('C:\\Users\\효원\\Desktop\\데이터셋\\embeddings\\payment_embeddings_with_intent.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("질문 + 고객의도 임베딩 JSON 저장 완료!")
