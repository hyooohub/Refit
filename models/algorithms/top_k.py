from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch

# 1. 모델 로드
bi_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

# 2. 임베딩된 질문 데이터 로드
with open("C:\\Users\\효원\\Desktop\\데이터셋\\embeddings\\payment_embeddings.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)["questions"]

corpus_questions = [item["질문"] for item in embedded_data]
corpus_embeddings = torch.tensor([item["embedding"] for item in embedded_data])

# 3. 답변 데이터 로드 (질문 순서 동일 가정)
with open("C:\\Users\\효원\\Desktop\\데이터셋\\data\\payment_cleaned.json", "r", encoding="utf-8") as f:
    raw_answers = json.load(f)
corpus_answers = [item["답변"] for item in raw_answers]

# 4. 사용자 질문 입력
user_question = "취소하고 싶어요"
user_embedding = bi_encoder.encode(user_question, convert_to_tensor=True)

# 5. Bi-Encoder로 Top-K 후보 추출
top_k = 100
cos_scores = util.cos_sim(user_embedding, corpus_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

# 6. CrossEncoder로 재정렬
cross_inp = [[user_question, corpus_questions[idx]] for idx in top_results.indices]
cross_scores = cross_encoder.predict(cross_inp)

# 7. CrossEncoder 상위 10개 후보 출력 (질문 + 답변)
print("\n🎯 [CrossEncoder 상위 10개 후보]")
for rank in range(10):
    idx = top_results.indices[rank]
    score = cross_scores[rank]
    
    if score < 0.8:  # 점수가 0.8 미만인 경우 "답변 불가" 처리
        print(f"- 질문: {corpus_questions[idx]}\n  ↳ 답변: 답변 불가\n  🧠 점수: {score:.4f}")
    else:
        answer = corpus_answers[idx].strip()
        print(f"- 질문: {corpus_questions[idx]}\n  ↳ 답변: {answer}\n  🧠 점수: {score:.4f}")

# 8. 실제 답변이 존재하는 최적 질문 선택
best_question = None
best_answer = None
for i in range(len(cross_scores)):
    idx = top_results.indices[i].item()
    score = cross_scores[i]
    answer = corpus_answers[idx].strip()
    
    if score >= 0.8 and answer and answer != "답변 없음":
        best_question = corpus_questions[idx]
        best_answer = answer
        break

# 9. 최종 결과 출력
print("\n✅ [최종 선택된 질문과 답변]")
if best_question and best_answer:
    print(f"💬 질문: {best_question}")
    print(f"✅ 답변: {best_answer}")
else:
    print("⚠️ 답변 가능한 유사 질문이 없습니다.")
