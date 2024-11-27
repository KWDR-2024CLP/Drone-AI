import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Yolov8 모델 로드
model = YOLO('C:/Users/user/Documents/GitHub/Drone-AI/YGNDR.pt')

# 모델 정보 출력
print(model)

def detect_saved_video(video_path):
    # 저장된 영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: '{video_path}' 파일을 열 수 없습니다. 경로를 확인하세요.")
        return

    # 비디오 저장 설정
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)  # 결과물 폴더 생성
    output_path = os.path.join(output_dir, "output.mp4")

    # 비디오 저장을 위한 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 형식 코덱
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 원본 영상의 FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 원본 영상의 가로 해상도
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 원본 영상의 세로 해상도
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상 탐지가 완료되었습니다.")
            break

        # Yolov8 모델로 프레임 감지
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
            scores = result.boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = result.boxes.cls.cpu().numpy()  # 클래스 ID

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 탐지 결과를 비디오 파일에 저장
        out.write(frame)

        # 탐지 결과 화면에 출력
        cv2.imshow("Saved Video Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"탐지 결과가 {output_path}에 저장되었습니다.")

# 사용 예시
video_path = "Sample/5.mp4"  # 드론 비디오 경로
detect_saved_video(video_path)
