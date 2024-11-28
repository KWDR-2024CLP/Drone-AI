import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load Yolov8 model
model = YOLO('/home/admin01/Drone-AI/YGNDR.pt')  # Update to the correct path

# Print model info
print(model)

def detect_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera index for the first camera

    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Perform object detection with the Yolov8 model
        results = model(frame)

        # Process results and draw bounding boxes
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame with detection results
        cv2.imshow("Webcam Feed - Object Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

# Call the function to start webcam detection
if __name__ == "__main__":
    detect_webcam()
