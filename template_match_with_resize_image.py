from cProfile import label
from cgitb import text
from copyreg import pickle
from ctypes import resize
from email.mime import image
from operator import index
import string
from tkinter.tix import IMAGE
from turtle import bgcolor, left
from typing import Text
import cv2
from cv2 import circle
import numpy as np
from PIL import Image
from PIL import ImageTk
from PIL import *
from tkinter import HORIZONTAL, VERTICAL, Button, Scale, Tk,Label,LabelFrame,StringVar
from setuptools import Command
from pickle import *
import cv2 
#import pytesseract
from tkinter import Entry
from tkinter import Spinbox
import numpy as np
import RPi.GPIO as GPIO
import time
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

OK_signal = 16
NOT_OK_signal = 20
Trigger = 19


GPIO.setmode(GPIO.BCM)
GPIO.setup(OK_signal, GPIO.OUT)
GPIO.setup(Trigger, GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(NOT_OK_signal, GPIO.OUT)

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(-1)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
def remove_noise(image):
    return cv2.medianBlur(image,5)
root=Tk()
root.geometry("700x900")
root.configure(bg="white")
Label(root,text="LIVE",bg="white",fg="black",font=("times new roman",25,"bold")).place(x=200,y=300)
f1=LabelFrame(root,bg="black")
f1.place(x=40,y=0)
live=Label(f1,bg="black")
live.pack()

Label(root,text="CAPTURED",bg="white",fg="black",font=("times new roman",25,"bold")).place(x=650,y=300)
f2=LabelFrame(root,bg="black")
f2.place(x=550,y=0)
live1=Label(f2,bg="black")
live1.pack()

Label(root,text="TEMPLATE",bg="white",fg="black",font=("times new roman",25,"bold")).place(x=1150,y=300)
f3=LabelFrame(root,bg="black")
f3.place(x=1050,y=0)
live2=Label(f3,bg="black")
live2.pack()

Label(root,text="RESULT",bg="white",fg="black",font=("times new roman",25,"bold")).place(x=680,y=750)
f4=LabelFrame(root,bg="black")
f4.place(x=550,y=370)
live3=Label(f4,bg="black")
live3.pack()

Label(root,text="Match_percentage",bg="white",fg="black",font=("times new roman",14,"bold")).place(x=10,y=570)
f5=LabelFrame(root,bg="black")
f5.place(x=10,y=500)

#inputtxt = Entry(root)
#inputtxt.place(x=10,y=600)
match_percent=StringVar(root)
match_percent.set("80")
sp = Spinbox(root, from_= 0, to = 100,textvariable=match_percent)
sp.place(x=10,y=600)

def resize_image(event):
    new_width = int(event.width * 0.50)
    new_height = int(event.height * 0.50)
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image = photo)
    label.image = photo #avoid garbage collection

image = Image.open('Result.jpg')
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(image)
label = ttk.Label(root, image = photo)
label.bind('<Configure>', resize_image)
#label.pack(fill=BOTH, expand = YES)

def find_match():
    img_rgb = cv2.imread('saved_img.jpg')
    template = cv2.imread('saved_img1.jpg', 1)
    #cv2.imshow("test",template)
    #cv2.waitKey(0)
    # Store width and height of template in w and h
    w, h = template.shape[:-1]
 
# Perform match operations.
    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
    thresh=sp.get()
    thresh=int(thresh)
    thresh=thresh/100
    #print(thresh)
    threshold = thresh
 
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
        GPIO.output(OK_signal, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(OK_signal, GPIO.LOW)
        print ((pt[0] + w, pt[1] + h))
        break

# Show the final image with the matched area.

    if flag==False:
        start_point = (230,53)
        end_point = (139,137)
        color = (0, 0, 255)
        thickness = 4
        cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
        GPIO.output(NOT_OK_signal, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(NOT_OK_signal, GPIO.LOW)
    #cv2.imshow('Detected', img_rgb)
    #cv2.waitKey(0)
    cv2.imwrite(filename='Result.jpg', img=img_rgb)


def template_capture():
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


def capture_image():
    image=Image.fromarray(img2)
    image.save("saved_img.jpg")

b1=Button(root,text="Capture",font=("times new roman",18,"bold"))
b1.config(command=capture_image)
b1.place(x=10,y=330)

b2=Button(root,text="Match",font=("times new roman",14,"bold"))
b2.config(command=find_match)
b2.place(x=10,y=470)

b3=Button(root,text="Template_capture",font=("times new roman",14,"bold"))
b3.config(command=template_capture)
b3.place(x=10,y=400)

def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    if brightness != 0:
        if brightness > 0:
          shadow = brightness
          max = 255
        else:
            shadow = 0
            max = 255 + brightness
            al_pha = (max - shadow) / 255
            ga_mma = shadow

		# The function addWeighted
		# calculates the weighted sum
		# of two a
        cal = cv2.addWeighted(img, al_pha,img, 0, ga_mma)
    else:
        cal = img
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
		# The function addWeighted calculates
		# the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,cal, 0, Gamma)

	# putText renders the specified
	# text string in the image.
    #cv2.putText(cal, 'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
    return cal

w1=Scale(root,from_=0,to=255,orient=HORIZONTAL,cursor="circle")
w1.set(255)
w1.place(x=200,y=700)
w2=Scale(root,from_=0,to=255,orient=HORIZONTAL,cursor="circle")
w2.set(128)
w2.place(x=200,y=750)



while True:
    brightness_value=int(w1.get())
    contrast_value=int(w2.get())
    # kernal=slider3.get()
    # rang=slider4.get()
    img=webcam.read()[1]
    img=controller(img,brightness_value,contrast_value)
    img3=cv2.detailEnhance(img,255,0.1)
    img=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    img2=img
    img=cv2.resize(img,(400,300))
    img1=Image.fromarray(img)
    img1=ImageTk.PhotoImage(img1)
    live["image"]=img1
    capture_img=cv2.imread("saved_img.jpg")
    capture_img=cv2.cvtColor(capture_img,cv2.COLOR_BGR2RGB)
    capture_img=cv2.resize(capture_img,(400,300))
    capture_img1=Image.fromarray(capture_img)
    capture_img2=ImageTk.PhotoImage(capture_img1)
    live1["image"]=capture_img2

    template_img=cv2.imread("saved_img1.jpg")
    template_img=cv2.cvtColor(template_img,cv2.COLOR_BGR2RGB)
    template_img=cv2.resize(template_img,(400,300))
    template_img1=Image.fromarray(template_img)
    template_img2=ImageTk.PhotoImage(template_img1)
    live2["image"]=template_img2

    result_img=cv2.imread("Result.jpg")
    result_img=cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
    result_img=cv2.resize(result_img,(400,380))
    result_img1=Image.fromarray(result_img,mode=None)
    result_img2=ImageTk.PhotoImage(result_img1)
    live3["image"]=result_img2
    root.update()
    print(GPIO.input(Trigger))
    if GPIO.input(Trigger)== 1:
        capture_image()
        cv2.waitKey(500)
        find_match()

    