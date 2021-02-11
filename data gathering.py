import cv2

cam = cv2.VideoCapture(0)
currentframe = 0

while (True):
    ret, frame = cam.read()
    name = './data/neutral/nu' + str(currentframe) + '.jpg'
    cv2.imwrite(name, frame)
    currentframe += 1
    if currentframe>380:
        break
    cv2.imshow('img', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
print ("done")
cam.release()
cv2.destroyAllWindows()
print()
