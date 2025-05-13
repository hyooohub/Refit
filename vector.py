from sentence_transformers import SentenceTransformer
import json

# 모델 로드
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 예시 데이터셋
with open("C:\\Users\\효원\\Desktop\\데이터셋\\data\\shipping_cleaned.json", "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# 임베딩된 데이터를 저장할 리스트
processed_data = []

# 데이터셋에서 질문을 임베딩
for item in cleaned_data:  # 여기서 'f'가 아닌 'cleaned_data'를 순회해야 합니다
    question = item["질문"]
    embedding = model.encode(question).tolist()  # 벡터를 리스트로 변환
    processed_data.append({
        "질문": question,
        "embedding": embedding
    })

# 결과를 JSON 파일로 저장
output_data = {
    "questions": processed_data
}

# JSON 파일로 저장
with open('C:\\Users\\효원\\Desktop\\데이터셋\\embeddings\\shipping_embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("JSON 파일로 저장 완료!")
