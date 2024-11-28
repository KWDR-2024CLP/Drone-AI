#### 고화질 영상 480p 로 바꾸기 
# ( ffmpeg 미리 다운 받아야함)
# https://kminito.tistory.com/108 링크 참고

import subprocess

def resize_with_ffmpeg(input_path, output_path):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', 'scale=trunc(iw/2)*2:480',  # 가로 해상도를 2의 배수로 조정
        '-c:v', 'libx264',  # 코덱 설정
        '-preset', 'fast',  # 인코딩 속도
        output_path
    ]
    subprocess.run(command)

# 사용 예시
input_video = "D:/Sample/sample.mp4"
output_video = "D:/output_video_2_480p.mp4"
resize_with_ffmpeg(input_video, output_video)
