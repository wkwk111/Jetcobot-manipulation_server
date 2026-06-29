#!/usr/bin/env python3
# encoding: utf-8
"""노트북 로컬 YOLO 검출 + 3D 파지계획 FastAPI 서비스.

라즈베리파이는 SSH 터널 없이 노트북의 LAN IP로 프레임과 현재 Flange pose를
전송한다. ``/detect``, ``/grasp-plan``, ``/v1/grasp-plan`` 요청 형식과
응답 형식은 기존 원격 딥러닝 서버와 호환된다.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO

from .geometry import Calibration, PlanError, compute_grasp_plan, load_calibration
from .settings import settings


# ============================================================
# 1. 응답 형식: 기존 /detect API 호환
# ============================================================

class Detection(BaseModel):
    label: str
    class_id: int
    confidence: float
    bbox: list[float]      # [x1, y1, x2, y2]
    center: list[float]    # [u, v]
    width: float
    height: float


class InferResponse(BaseModel):
    status: str
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]
    saved_dir: Optional[str] = None
    raw_image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    result_json_path: Optional[str] = None


# ============================================================
# 2. 서버 상태
# ============================================================

class ServerState:
    model: YOLO
    calibration: Calibration
    inference_lock: Lock


state = ServerState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.euler_order != "zyx":
        raise RuntimeError("This project currently supports only EULER_ORDER=zyx")
    if not settings.model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {settings.model_path}")

    # 모델과 calibration은 서버 시작 시 한 번만 로드합니다.
    state.model = YOLO(str(settings.model_path))
    state.calibration = load_calibration(settings)
    state.inference_lock = Lock()

    print("[STARTUP] YOLO model:", settings.model_path)
    print("[STARTUP] device:", settings.device)
    print("[STARTUP] intrinsic:", settings.intrinsic_file)
    print("[STARTUP] hand-eye:", settings.handeye_result_json)
    print("[STARTUP] hand-eye method:", state.calibration.selected_method)
    yield


app = FastAPI(title="Laptop YOLO + Robot Grasp Planning Service", version="2.1.0", lifespan=lifespan)


# ============================================================
# 3. 기본 API
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": "laptop-local",
        "model_path": str(settings.model_path),
        "device": settings.device,
        "default_conf": settings.default_conf,
        "default_target_label": settings.default_target_label,
        "euler_order": settings.euler_order,
        "calibration_method": state.calibration.selected_method,
    }


# ============================================================
# 4. 공통 유틸
# ============================================================

def _dump(model: BaseModel) -> dict[str, Any]:
    """Pydantic v1/v2 양쪽에서 동작하도록 직렬화."""
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def decode_image(file_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("이미지 디코딩 실패")
    return image


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
        u, v = [int(round(value)) for value in det.center]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{det.label} {det.confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(annotated, (u, v), 4, (0, 0, 255), -1)
    return annotated


def save_detection_result(
    *,
    frame: np.ndarray,
    annotated: np.ndarray,
    detections: list[Detection],
    image_width: int,
    image_height: int,
    inference_ms: float,
    conf: float,
    imgsz: int,
    target_label: Optional[str],
    extra_result: dict[str, Any] | None = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """기존 서버의 이미지/JSON 로그 기능을 유지합니다."""
    if not settings.save_results:
        return None, None, None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = settings.save_root_dir / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_path = save_dir / "raw.jpg"
    annotated_path = save_dir / "annotated.jpg"
    result_path = save_dir / "result.json"

    cv2.imwrite(str(raw_path), frame)
    cv2.imwrite(str(annotated_path), annotated)

    result_data: dict[str, Any] = {
        "timestamp": timestamp,
        "model_path": str(settings.model_path),
        "device": settings.device,
        "request": {"conf": conf, "imgsz": imgsz, "target_label": target_label},
        "image": {
            "width": image_width,
            "height": image_height,
            "raw_image_path": str(raw_path),
            "annotated_image_path": str(annotated_path),
        },
        "inference_ms": inference_ms,
        "detections": [_dump(det) for det in detections],
    }
    if extra_result is not None:
        result_data["grasp_plan_result"] = extra_result

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return str(save_dir), str(raw_path), str(annotated_path), str(result_path)


def _validate_frame_size(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    if settings.expected_image_width and w != settings.expected_image_width:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image width {w} differs from EXPECTED_IMAGE_WIDTH "
                f"{settings.expected_image_width}; camera calibration may be invalid"
            ),
        )
    if settings.expected_image_height and h != settings.expected_image_height:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image height {h} differs from EXPECTED_IMAGE_HEIGHT "
                f"{settings.expected_image_height}; camera calibration may be invalid"
            ),
        )


def _effective_target_label(requested: str | None) -> str | None:
    # API query가 주어지면 그 값을 사용하되, 없으면 서버 정책의 기본 클래스를 사용합니다.
    return requested if requested is not None else settings.default_target_label


def _run_inference(
    *,
    frame: np.ndarray,
    conf: float,
    imgsz: int,
    target_label: str | None,
) -> tuple[list[Detection], float]:
    start = time.perf_counter()
    # 단일 GPU 모델은 요청 간 동시 접근을 막아 메모리/결과 충돌을 피합니다.
    with state.inference_lock:
        results = state.model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=settings.device,
            verbose=False,
        )
    inference_ms = (time.perf_counter() - start) * 1000.0

    detections: list[Detection] = []
    if not results or results[0].boxes is None:
        return detections, inference_ms

    result = results[0]
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(float)
        score = float(box.conf[0].detach().cpu().item())
        class_id = int(box.cls[0].detach().cpu().item())
        label = str(names.get(class_id, class_id))
        if target_label is not None and label != target_label:
            continue

        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        detections.append(
            Detection(
                label=label,
                class_id=class_id,
                confidence=score,
                bbox=[x1, y1, x2, y2],
                center=[u, v],
                width=x2 - x1,
                height=y2 - y1,
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections, inference_ms


def _read_and_decode_upload(image: UploadFile, raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="빈 이미지 업로드입니다.")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="업로드 이미지가 MAX_UPLOAD_BYTES를 초과했습니다.")
    try:
        frame = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _validate_frame_size(frame)
    return frame


def _parse_robot_state(raw: str) -> tuple[list[float], str | None]:
    """Pi가 보내는 `robot_state` form field를 검증합니다.

    형식: {"request_id":"...", "flange_coords":[x,y,z,rx,ry,rz]}
    """
    try:
        robot_state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="robot_state는 JSON 문자열이어야 합니다.") from exc

    coords = robot_state.get("flange_coords") if isinstance(robot_state, dict) else None
    if not isinstance(coords, list) or len(coords) != 6:
        raise HTTPException(status_code=400, detail="robot_state.flange_coords는 6개 숫자여야 합니다.")
    try:
        parsed = [float(value) for value in coords]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="flange_coords에 숫자가 아닌 값이 있습니다.") from exc
    if not np.isfinite(np.asarray(parsed, dtype=np.float64)).all():
        raise HTTPException(status_code=400, detail="flange_coords에 유한하지 않은 값이 있습니다.")

    request_id = robot_state.get("request_id")
    return parsed, str(request_id) if request_id is not None else None


# ============================================================
# 5. 기존 YOLO detect API (호환 유지)
# ============================================================

@app.post("/detect", response_model=InferResponse)
async def detect(
    image: UploadFile = File(...),
    conf: float = Query(settings.default_conf, ge=0.0, le=1.0),
    imgsz: int = Query(settings.default_imgsz, ge=32),
    target_label: Optional[str] = Query(None),
) -> InferResponse:
    raw = await image.read()
    frame = _read_and_decode_upload(image, raw)
    h, w = frame.shape[:2]
    effective_target_label = _effective_target_label(target_label)

    try:
        detections, inference_ms = _run_inference(
            frame=frame,
            conf=conf,
            imgsz=imgsz,
            target_label=effective_target_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {exc}") from exc

    annotated = draw_detections(frame, detections)
    save_dir, raw_path, annotated_path, json_path = save_detection_result(
        frame=frame,
        annotated=annotated,
        detections=detections,
        image_width=w,
        image_height=h,
        inference_ms=inference_ms,
        conf=conf,
        imgsz=imgsz,
        target_label=effective_target_label,
    )
    print("[DETECT] raw:", raw_path)
    print("[DETECT] annotated:", annotated_path)
    print("[DETECT] json:", json_path)

    return InferResponse(
        status="ok",
        image_width=w,
        image_height=h,
        inference_ms=inference_ms,
        detections=detections,
        saved_dir=save_dir,
        raw_image_path=raw_path,
        annotated_image_path=annotated_path,
        result_json_path=json_path,
    )


# ============================================================
# 6. 서버측 YOLO + 3D 파지계획 API
# ============================================================

async def _create_grasp_plan(
    *,
    image: UploadFile,
    robot_state: str,
) -> dict[str, Any]:
    """노트북의 고정 정책으로 파지 계획을 생성합니다.

    라즈베리파이는 frame과 촬영 시점의 현재 Flange pose만 보내며,
    YOLO 및 모든 좌표변환은 이 함수 내부에서 완료합니다.
    """
    current_flange_coords, request_id = _parse_robot_state(robot_state)
    raw = await image.read()
    frame = _read_and_decode_upload(image, raw)
    h, w = frame.shape[:2]

    try:
        detections, inference_ms = _run_inference(
            frame=frame,
            conf=settings.default_conf,
            imgsz=settings.default_imgsz,
            target_label=settings.default_target_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {exc}") from exc

    annotated = draw_detections(frame, detections)
    extra_result: dict[str, Any] = {
        "request_id": request_id,
        "current_flange_coords": current_flange_coords,
    }

    if not detections:
        extra_result.update({"status": "not_found", "message": "Target object was not detected."})
        save_dir, raw_path, annotated_path, json_path = save_detection_result(
            frame=frame,
            annotated=annotated,
            detections=detections,
            image_width=w,
            image_height=h,
            inference_ms=inference_ms,
            conf=settings.default_conf,
            imgsz=settings.default_imgsz,
            target_label=settings.default_target_label,
            extra_result=extra_result,
        )
        return {
            "status": "not_found",
            "request_id": request_id,
            "message": "No target detection satisfied the server target-label/confidence policy.",
            "image_width": w,
            "image_height": h,
            "inference_ms": inference_ms,
            "detections": [_dump(det) for det in detections],
            "saved_dir": save_dir,
            "raw_image_path": raw_path,
            "annotated_image_path": annotated_path,
            "result_json_path": json_path,
        }

    # detections는 confidence 내림차순. 가장 높은 대상 하나를 서버가 선택합니다.
    selected = detections[0]
    try:
        result = compute_grasp_plan(
            detection=_dump(selected),
            current_flange_coords=current_flange_coords,
            calibration=state.calibration,
            settings=settings,
        )
    except PlanError as exc:
        raise HTTPException(status_code=422, detail=f"Grasp-plan geometry error: {exc}") from exc

    extra_result.update({"status": "ok", **result})
    save_dir, raw_path, annotated_path, json_path = save_detection_result(
        frame=frame,
        annotated=annotated,
        detections=detections,
        image_width=w,
        image_height=h,
        inference_ms=inference_ms,
        conf=settings.default_conf,
        imgsz=settings.default_imgsz,
        target_label=settings.default_target_label,
        extra_result=extra_result,
    )

    return {
        "status": "ok",
        "request_id": request_id,
        "image_width": w,
        "image_height": h,
        "inference_ms": inference_ms,
        "detections": [_dump(det) for det in detections],
        "server_policy": {
            "target_label": settings.default_target_label,
            "confidence_threshold": settings.default_conf,
            "object_plane_z_base_mm": settings.object_plane_z_base_mm,
            "target_z_offset_mm": settings.toothbrush_target_z_offset_mm,
        },
        **result,
        "saved_dir": save_dir,
        "raw_image_path": raw_path,
        "annotated_image_path": annotated_path,
        "result_json_path": json_path,
    }


@app.post("/grasp-plan")
async def grasp_plan(
    image: UploadFile = File(...),
    robot_state: str = Form(...),
) -> dict[str, Any]:
    return await _create_grasp_plan(image=image, robot_state=robot_state)


# 이전 분리형 클라이언트와 호환되는 alias입니다.
@app.post("/v1/grasp-plan")
async def grasp_plan_v1(
    image: UploadFile = File(...),
    robot_state: str = Form(...),
) -> dict[str, Any]:
    return await _create_grasp_plan(image=image, robot_state=robot_state)
