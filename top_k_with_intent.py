from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch

# 1. 모델 로드
bi_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

# 2. 임베딩된 질문 + 고객의도 데이터 로드
with open("C:\\Users\\효원\\Desktop\\데이터셋\\embeddings\\payment_embeddings_with_intent.json", "r", encoding="utf-8") as f:
    embedded_data = json.load(f)["questions"]

corpus_questions = [item["질문"] for item in embedded_data]
corpus_embeddings = torch.tensor([item["embedding"] for item in embedded_data])
corpus_intent_embeddings = torch.tensor([item["intent_embedding"] for item in embedded_data])

# 3. 답변 로드 (질문 순서 동일 가정)
with open("C:\\Users\\효원\\Desktop\\데이터셋\\data\\payment_cleaned_with_intent.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

corpus_answers = [item.get("답변", "") for item in raw_data]

# 4. 사용자 질문 입력
user_question = "다른 카드로 결제 변경하고 싶어요"
f
# 5. 사용자 질문 임베딩
user_embedding = bi_encoder.encode(user_question, convert_to_tensor=True)

# 6. 사용자 의도 벡터 == 질문 벡터로 임시 대체 (고객의도 분류 모델이 없을 경우)
user_intent_embedding = user_embedding  # <- 분리된 intent 모델 결과 없으므로 동일 벡터 사용

# 7. 질문 + 고객의도 유사도 계산
cos_scores_q = util.cos_sim(user_embedding, corpus_embeddings)[0]
cos_scores_intent = util.cos_sim(user_intent_embedding, corpus_intent_embeddings)[0]

# 8. 최종 유사도 점수 (가중치 조절 가능)
final_scores = 0.8 * cos_scores_q + 0.2 * cos_scores_intent
top_k = 100
top_results = torch.topk(final_scores, k=top_k)

# 9. CrossEncoder 재정렬
cross_inp = [[user_question, corpus_questions[idx]] for idx in top_results.indices]
cross_scores = cross_encoder.predict(cross_inp)

# 10. CrossEncoder 점수를 기준으로 내림차순 정렬
sorted_scores_idx = torch.argsort(torch.tensor(cross_scores), descending=True)

# 11. 상위 10개 출력 (답변 불가 처리는 하지 않음)
print("\n🎯 [CrossEncoder 상위 10개 후보]")
for rank in range(10):
    idx = top_results.indices[sorted_scores_idx[rank]]
    score = cross_scores[sorted_scores_idx[rank]]
    answer = corpus_answers[idx].strip()

    print(f"- 질문: {corpus_questions[idx]}")
    print(f"  ↳ 답변: {answer}")
    print(f"  🧠 점수: {score:.4f}\n")

# 12. 최종 결과 선택 (가장 높은 점수 순으로)
best_question, best_answer = None, None
for rank in sorted_scores_idx:
    idx = top_results.indices[rank].item()
    score = cross_scores[rank].item()
    answer = corpus_answers[idx].strip()

    if score >= 0.75 and answer and answer != "답변 없음":
        best_question = corpus_questions[idx]
        best_answer = answer
        break

# 13. 최종 결과 출력
print("\n✅ [최종 선택된 질문과 답변]")
if best_question and best_answer:
    print(f"💬 질문: {best_question}")
    print(f"✅ 답변: {best_answer}")
else:
    print("⚠️ 답변 가능한 유사 질문이 없습니다.")
