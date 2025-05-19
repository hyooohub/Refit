#!/bin/bash
# ✅ KoGPT2 파인튜닝 모델 다운로드 스크립트
# Google Drive에서 모델을 다운로드합니다.

# 1. gdown 설치 (필요 시)
pip install gdown

# 2. 공유된 Google Drive 폴더 ID 설정 (링크의 /folders/ 뒷부분)
FOLDER_ID="https://drive.google.com/drive/folders/1dJllLLwKA6EVE9xqbeAn26D0cd_yzybm?usp=sharing"  # ✅ 실제 ID로 바꿔주세요

# 3. 다운로드 위치 지정
DEST_DIR="./models/KoGPT2/KoGPT2_finetuned2_final"

# 4. 다운로드 수행
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" --output "${DEST_DIR}"

# 5. 완료 메시지
echo "✅ 모델 다운로드 완료: ${DEST_DIR}"
