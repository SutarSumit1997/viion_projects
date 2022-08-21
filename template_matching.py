import cv2
import numpy as np
import RPi.GPIO as GPIO
import time


led = 20
led1 = 21
switch = 26
reset_sw=6

GPIO.setmode(GPIO.BCM)
GPIO.setup(led, GPIO.OUT)
GPIO.setup(switch, GPIO.IN)
GPIO.setup(reset_sw, GPIO.IN)
GPIO.setup(led1, GPIO.OUT)
GPIO.output(led1, GPIO.HIGH)
GPIO.output(led, GPIO.HIGH)
key = cv2. waitKey()
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
        print('Switch status = ', GPIO.input(switch))
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
# Read the main image
print("processing")
img_rgb = cv2.imread('saved_img.jpg')
print("reding img")
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# Read the template
template = cv2.imread('saved_img1.jpg', 0)
print("reding_template")
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
print("matching_template")
# Specify a threshold
threshold = 0.50
 
# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)
#print(str(loc))
# Draw a rectangle around the matched region.
flag=False
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
    #print(pt)
    #print((pt[0] + w, pt[1] + h))
    flag=True
    GPIO.output(led1, GPIO.LOW)
    print("OUTPUT_SET")
    #print ((pt[0] + w, pt[1] + h))

# Show the final image with the matched area.

if flag==False:
    start_point = (240,186)
    end_point = (528,311)
    color = (0, 0, 255)
    thickness = 4
    cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
    cv2.imshow('Detected', img_rgb)
    print('reset_Switch status = ', GPIO.input(reset_sw))
    cv2.waitKey(5000)
    

#Reseting
print("WAITING_FOR_RESET")
print('reset_Switch status = ', GPIO.input(reset_sw))
if GPIO.input(reset_sw) == 0: 
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()


GPIO.output(led1, GPIO.HIGH)
GPIO.cleanup()
