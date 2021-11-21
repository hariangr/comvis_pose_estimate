import cv2

camindex = 0
vcap = cv2.VideoCapture(camindex)

if not vcap.isOpened():
    raise Exception(f"Failed to open camera at index {camindex}")

while True:
    ret, frame = vcap.read()
    
    cv2.imshow("Raw Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break