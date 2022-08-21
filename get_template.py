import cv2
import pytesseract
import RPi.GPIO as GPIO
import time

led = 20
led1 = 21
switch = 26


GPIO.setmode(GPIO.BCM)
GPIO.setup(led, GPIO.OUT)
GPIO.setup(switch, GPIO.IN)
GPIO.setup(led1, GPIO.OUT)
GPIO.output(led1, GPIO.LOW)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
def remove_noise(image):
    return cv2.medianBlur(image,5)
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
       # print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if GPIO.input(switch) == 1:
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.COLOR_BAYER_GR2BGR)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
        
img_raw = cv2.imread("saved_img.jpg")


#select ROI function
roi = cv2.selectROI(img_raw)

#print rectangle points of selected roi
print(roi)

#Crop selected roi from raw image
roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
print(int(roi[1]))
print(int(roi[1]+roi[3]))
print(int(roi[0]))
print(int(roi[0]+roi[2]))

cv2.imwrite(filename='saved_img1.jpg', img=roi_cropped)

cv2.destroyAllWindows()