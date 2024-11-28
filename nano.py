import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load Yolov8 model
model = YOLO('/home/admin01/Drone-AI/YGNDR.pt')  # Update to the correct path

# Print model info
print(model)

def detect_saved_video(video_path):
    # Open the saved video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'. Check the path.")
        return

    # Video saving settings
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)  # Create result folder
    output_path = os.path.join(output_dir, "output.mp4")

    # Video writer settings for saving the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Original video's FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Original video's width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Original video's height
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video detection completed.")
            break

        # Detect with Yolov8 model
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save detection results to video file
        out.write(frame)

        # Display detection results on the screen
        cv2.imshow("Saved Video Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Detection results saved to {output_path}.")

# Example usage
video_path = "/home/admin01/Drone-AI/Sample/5.mp4"  # Update to the correct path
detect_saved_video(video_path)
