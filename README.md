# Jetcobot-manipulation_server

## 1. 노트북 준비

`models/best.pt`, `calibration/`이 포함된 이 폴더 전체를 노트북에 둡니다.

### Ubuntu Terminal

```mkdir dl_server
cd dl_server
python3 -m venv ~/venv/dl_server
source ~/venv/dl_server/bin/activate
pip install -r requirements.txt
```

## 2. Yolo 모델 파라미터 준비

`models/`에 download 받거나 학습한 모델 가중치를 업로드하여 `models/best.pt`가 위치하게합니다.

https://drive.google.com/file/d/1jt9kHi6C1yFN-McUetMwOg-4TwDhj613/view?usp=sharing

## 3. 딥러닝 서버 실행

`run_cilent.py`를 실행하여 Jetcobot의 이미지를 처리할 준비를 합니다.

### Ubuntu Terminal

```cd dl_server
source ~/venv/dl_server/bin/activate
python3 run_server.py
```

1. 8000번 포트를 이용하기 때문에 8000번 포트를 이용하는 프로세스는 종료해주세요
2. `models/best.pt`가 있는지 확인해 주세요