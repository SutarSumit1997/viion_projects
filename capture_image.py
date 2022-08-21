import cv2 
import pytesseract
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
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img_new.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img_new.jpg', cv2.COLOR_BAYER_GR2BGR)
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
image=cv2.imread("ufi1.jpg") 
# Crop image
#cropped_image = image[438:644,417:747]
# Display cropped image
#cv2.imshow("Cropped image", cropped_image)
cv2.waitKey(0)
text = pytesseract.image_to_string(cropped_image)
print(text)
cv2.imshow("Frame1", cropped_image)
cv2.waitKey(0)
print("done")