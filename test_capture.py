import cv2

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,200)
cap.set(cv2.CAP_PROP_EXPOSURE, -3.5)
cap.set(cv2.CAP_PROP_FPS,3000)
cap.set(cv2.CAP_PROP_ZOOM,0)
def thresholding(image):
    return cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)


while True:
    ret, img = cap.read()
    cv2.imshow("input", img)
    gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    gray_img=gray_img
    x1=200
    x2=100
    y1=140
    y2=50
    gray_img = gray_img[y1:y1+y2,x1:x1+x2]
    gray_img=thresholding(gray_img)[1]
    cv2.imshow("test_gray",gray_img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()