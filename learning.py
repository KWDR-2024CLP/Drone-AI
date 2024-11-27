import torch
from ultralytics import YOLO

def train_yolo_model():
    # YOLOv8n 모델 로드
    model = YOLO('yolov8n.pt')

    # 데이터 경로 및 학습 설정
    model.train(
        data='D:/DRONE.v2i.yolov8/data.yaml',  # 데이터셋 YAML 파일 경로
        epochs=50,  # 최대 학습 에포크 수
        batch=16,  # 배치 크기
        imgsz=640,  # 입력 이미지 크기
        project='runs/detect',  # 결과 저장 경로
        name='drone_detection',  # 프로젝트 이름
        device=0,  # CUDA 장치
        patience=10  # 조기 종료: 10 에포크 동안 개선되지 않으면 종료
    )

if __name__ == "__main__":
    train_yolo_model()
