import cv2

import mediapipe as mp
import time

USE_CAMERA = True


def main():
    # region Source of Video
    if USE_CAMERA:
        camindex = 0
        cap = cv2.VideoCapture(camindex)
        if not cap.isOpened():
            raise Exception(f"Failed to open camera at index {camindex}")
    else:
        cap = cv2.VideoCapture('./sample/a.mp4')
    # endregion

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    while True:
        ok, frameRaw = cap.read()
        if not ok:
            raise Exception("Failed to read frame")

        frameCanvas = frameRaw.copy()

        frameRGB = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(frameCanvas, results.pose_landmarks,
                                  mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frameCanvas.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frameCanvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Raw Frame", frameRaw)
        cv2.imshow("Image", frameCanvas)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
