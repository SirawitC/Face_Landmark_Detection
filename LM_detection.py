import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10)
draw_spec = mpDraw.DrawingSpec(thickness=1,circle_radius=1, color = (0,255,0))
while True:
    succ, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for fLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, fLms, mpFaceMesh.FACEMESH_CONTOURS,draw_spec,draw_spec)
            for id,lm in enumerate(fLms.landmark):
                h, w, c = img.shape
                x,y = int(lm.x*w), int(lm.y*h)
                print(id,x,y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f"FPS: {int(fps)}",(20,70),cv2.FONT_HERSHEY_PLAIN,
                3,(0,255,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

