import torch
import cv2
import numpy as np

# YOLO 모델 로드 (최적화)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s/weights/best.pt')

def detect_saved_video(video_path):
    # 저장된 영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: '{video_path}' 파일을 열 수 없습니다. 경로를 확인하세요.")
        return

    frame_jump = 10  # 10프레임씩 이동
    print(f"10프레임 단위로 이동 설정 완료")

    is_paused = False  # 일시정지 상태를 추적하는 변수

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("영상 탐지가 완료되었습니다.")
                break

            # YOLO 모델을 사용하여 탐지
            results = model(frame)
            result_frame = np.squeeze(results.render())

            # 화면 크기 조정 (50% 크기로 축소)
            resized_frame = cv2.resize(result_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # 탐지 결과 출력
            cv2.imshow("Saved Video Detection", resized_frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # 'q' 키를 누르면 종료
            print("탐지를 종료합니다.")
            break
        elif key == 32:  # 스페이스바: 일시정지/재생 전환
            is_paused = not is_paused
        elif key == ord('d'):  # D 키: +10프레임 앞으로 이동
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            new_frame = min(total_frames - 1, current_frame + frame_jump)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"+10프레임 이동: {new_frame}/{total_frames} 프레임")
        elif key == ord('a'):  # A 키: -10프레임 뒤로 이동
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - frame_jump)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"-10프레임 이동: {new_frame}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} 프레임")

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 사용 예시
video_path = "Y:/rapid_flight.mp4"  # 저장된 영상 파일 경로
detect_saved_video(video_path)
