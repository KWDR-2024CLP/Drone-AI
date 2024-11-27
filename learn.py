from ultralytics import YOLO

# YOLOv8 모델 생성 또는 기존 모델 로드
# 새로 학습하려면 YOLOv8 모델 아키텍처를 불러옵니다.
model = YOLO("yolov8n.pt")  # 'yolov8n.pt'는 YOLOv8의 기본 네트워크 구조입니다.

# 데이터 경로 및 설정 파일 설정
data_yaml_path = "Y:/DRONE.v2i.yolov8/data.yaml"  # 데이터셋의 YAML 파일 경로
epochs = 50  # 학습 반복 횟수
batch_size = 16  # 배치 크기

# 모델 학습
model.train(
    data=data_yaml_path,  # 데이터셋 설정 파일 경로
    epochs=epochs,  # 학습 반복 횟수
    batch=batch_size,  # 배치 크기
    imgsz=640,  # 이미지 크기
    name="yolov8_person_detection",  # 결과 저장 폴더 이름
    workers=4,  # 데이터 로딩 워커 수
    device=0,  # 학습에 사용할 GPU (0: 첫 번째 GPU, "cpu": CPU)
)

# 학습 완료 후 최적의 가중치(best.pt) 저장 경로
best_model_path = "runs/detect/yolov8_person_detection/weights/best.pt"
print(f"최적의 가중치 파일 경로: {best_model_path}")

# 학습 결과 확인
model.val(data=data_yaml_path)  # 검증 수행
