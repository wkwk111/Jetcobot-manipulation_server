#!/usr/bin/env python3
"""라즈베리파이에서 노트북 로컬 추론 서비스를 호출하는 안전한 통신 예제.

이 예제는 로봇을 움직이지 않습니다. 반환된 flange_command를 출력만 하므로,
기존 Pi 제어 코드에서 HTTP URL과 응답 처리 부분을 교체할 때 참고하십시오.

예시:
  python3 examples/pi_http_client_example.py \
      --server http://192.168.0.20:8000 \
      --image /tmp/frame.jpg \
      --pose 196.26,92.43,219.19,-177.46,4.93,-94.65
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import requests


def parse_pose(raw: str) -> list[float]:
    try:
        pose = [float(value.strip()) for value in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose must be six comma-separated numbers") from exc
    if len(pose) != 6:
        raise argparse.ArgumentTypeError("pose must contain x,y,z,rx,ry,rz")
    return pose


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one frame to the laptop YOLO/grasp-plan service.")
    parser.add_argument("--server", required=True, help="e.g. http://192.168.0.20:8000")
    parser.add_argument("--image", required=True, type=Path, help="640x480 JPEG/PNG captured by the Pi camera")
    parser.add_argument("--pose", required=True, type=parse_pose, help="current flange x,y,z,rx,ry,rz in mm/deg")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")

    robot_state = {"request_id": str(uuid4()), "flange_coords": args.pose}
    endpoint = args.server.rstrip("/") + "/v1/grasp-plan"
    with args.image.open("rb") as image_file:
        response = requests.post(
            endpoint,
            files={"image": (args.image.name, image_file, "image/jpeg")},
            data={"robot_state": json.dumps(robot_state)},
            timeout=args.timeout,
        )
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("status") == "ok":
        print("\nflange_command for existing Pi robot-control code:")
        print(result["plan"]["flange_command"])
        return 0
    print("\nTarget was not found; do not move the robot.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
