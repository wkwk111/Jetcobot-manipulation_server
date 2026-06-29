# Jetcobot-manipulation_server

## 1. 노트북 준비

`models/best.pt`, `calibration/`이 포함된 이 폴더 전체를 노트북에 둡니다.

### Ubuntu Terminal

```mkdir dl_server
cd dl_server
python3 -m venv ~/venv/dl_server
pip install -r requirements.txt
```

## 2. Yolo 모델 파라미터 준비

`models/`에 download 받거나 학습한 모델 가중치를 업로드하여 `models/best.pt`가 위치하게합니다.

https://drive.google.com/file/d/1jt9kHi6C1yFN-McUetMwOg-4TwDhj613/view?usp=sharing