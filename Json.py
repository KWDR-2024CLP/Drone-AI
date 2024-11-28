## json 데이터 yolo 로 전환
import os
import json

# 경로 설정
json_dir = "path/to/json_files"  # AI Hub JSON 파일 폴더 경로
image_dir = "path/to/images"  # AI Hub 이미지 파일 폴더 경로
output_label_dir = "path/to/output_labels"  # 변환된 YOLO 라벨 폴더
os.makedirs(output_label_dir, exist_ok=True)

# 클래스 매핑 (수정 필요)
classes = ["person", "car", "bicycle"]  # JSON에서 사용된 클래스 이름

# JSON → YOLO 변환
for json_file in os.listdir(json_dir):
    if json_file.endswith(".json"):
        with open(os.path.join(json_dir, json_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # 이미지 크기
        image_width = data["imageWidth"]
        image_height = data["imageHeight"]

        yolo_labels = []

        for obj in data["shapes"]:
            class_name = obj["label"]
            if class_name in classes:
                class_id = classes.index(class_name)

                points = obj["points"]
                xmin, ymin = points[0]
                xmax, ymax = points[1]

                # YOLO 포맷 계산
                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

        # YOLO 라벨 저장
        label_file = os.path.join(output_label_dir, os.path.splitext(json_file)[0] + ".txt")
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_labels))

print("JSON 데이터를 YOLO 포맷으로 변환 완료!")
