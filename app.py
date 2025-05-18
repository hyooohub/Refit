from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# 모델 로드
bi_encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

# 카테고리별 데이터 캐시
loaded_data = {}

# 상대경로 기준 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings")
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_category_data(category):
    if category in loaded_data:
        return loaded_data[category]

    embedding_path = os.path.join(EMBEDDING_DIR, f"{category}_embeddings_with_intent.json")
    data_path = os.path.join(DATA_DIR, f"{category}_cleaned_with_intent.json")

    if not os.path.exists(embedding_path) or not os.path.exists(data_path):
        return None

    with open(embedding_path, "r", encoding="utf-8") as f:
        embedded_data = json.load(f)["questions"]

    with open(data_path, "r", encoding="utf-8") as f:
        cleaned_data = json.load(f)  # ← 여기 변수명 변경

    questions = [item["질문"] for item in embedded_data]
    answers_dict = {item["질문"]: item.get("답변", "") for item in cleaned_data}  # ← 여기도 반영

    data = {
        "questions": questions,
        "embeddings": torch.tensor([item["embedding"] for item in embedded_data]),
        "intent_embeddings": torch.tensor([item["intent_embedding"] for item in embedded_data]),
        "answers": [answers_dict.get(q, "") for q in questions],
    }

    loaded_data[category] = data
    return data


@app.route("/ask", methods=["POST"])
def ask():
    req = request.get_json()
    user_question = req.get("question", "").strip()
    category = req.get("category", "").strip()

    if not user_question or not category:
        return jsonify({"error": "question과 category를 모두 입력하세요."}), 400

    data = load_category_data(category)
    if not data:
        return jsonify({"error": f"'{category}' 카테고리에 해당하는 데이터를 찾을 수 없습니다."}), 404

    user_embedding = bi_encoder.encode(user_question, convert_to_tensor=True)
    user_intent_embedding = user_embedding  # intent 임시 대체

    cos_scores_q = util.cos_sim(user_embedding, data["embeddings"])[0]
    cos_scores_intent = util.cos_sim(user_intent_embedding, data["intent_embeddings"])[0]
    final_scores = 0.8 * cos_scores_q + 0.2 * cos_scores_intent

    top_k = 100
    top_results = torch.topk(final_scores, k=top_k)
    cross_inp = [[user_question, data["questions"][idx]] for idx in top_results.indices]
    cross_scores = cross_encoder.predict(cross_inp)
    sorted_scores_idx = torch.argsort(torch.tensor(cross_scores), descending=True)

    for rank in sorted_scores_idx:
        idx = top_results.indices[rank].item()
        score = cross_scores[rank].item()
        answer = data["answers"][idx].strip()

        if score >= 0.75 and answer and answer != "답변 없음":
            return jsonify({
                "matched_question": data["questions"][idx],
                "answer": answer,
                "score": round(score, 4)
            })

    return jsonify({"message": "답변 가능한 유사 질문이 없습니다."})

if __name__ == "__main__":
    app.run(debug=True)
