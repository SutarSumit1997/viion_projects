from cProfile import label
from cgitb import text
from copyreg import pickle
from ctypes import resize
from email.mime import image
from operator import index, truediv
import string
from tkinter.font import BOLD
from tkinter.tix import IMAGE
from turtle import bgcolor, left
from typing import Text, ValuesView
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
#import RPi.GPIO as GPIO
import time
from tkinter import Canvas


OK_signal = 16
NOT_OK_signal = 20
Trigger = 19

'''
GPIO.setmode(GPIO.BCM)
GPIO.setup(OK_signal, GPIO.OUT)
GPIO.setup(Trigger, GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(NOT_OK_signal, GPIO.OUT)
'''
bc_value_retain = open("bcvalues.txt","r") 
#print(bc_value_retain.read(5))
bc_value_string=bc_value_retain.read()
#print(bc_value_string)
brightness_value=bc_value_string[0:3]
contrast_value=bc_value_string[3:8]
print(brightness_value)
print(contrast_value)
bc_value_retain.close()
brightness_value=int(brightness_value)
contrast_value=int(contrast_value)
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
def remove_noise(image):
    return cv2.medianBlur(image,5)
root=Tk()
root.geometry("1200x1100")
root.configure(bg="white")
Label(root,text="LIVE",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=50,y=220)
f1=LabelFrame(root,bg="black")
f1.place(x=60,y=10)
live=Label(f1,bg="black")
live.pack()

my_canvas =Canvas(root, width=50, height=50,background="white")  # Create 200x200 Canvas widget
my_canvas.place(x=550,y=290)

my_oval = my_canvas.create_oval(5, 5, 50, 50,width=4)  # Create a circle on the Canvas
#my_oval1 = my_canvas.create_oval(5, 170, 150, 300,width=7)
outputText="READY"
var = StringVar()
l = Label(root, textvariable=var,wraplength=398,font=("times new roman",28,"bold"),bg="WHITE")
l.place(x=680,y=350)
var.set("")

my_canvas1 =Canvas(root, width=50, height=50,background="white")  # Create 200x200 Canvas widget
my_canvas1.place(x=550,y=355)

my_oval1 = my_canvas1.create_oval(5, 5, 50, 50,width=4)  # Create a circle on the Canvas
#my_oval1 = my_canvas.create_oval(5, 170, 150, 300,width=7)
outputText="READY"
var = StringVar()
l = Label(root, textvariable=var,wraplength=398,font=("times new roman",28,"bold"),bg="WHITE")
l.place(x=680,y=350)
var.set("")

my_canvas2 =Canvas(root, width=50, height=50,background="white")  # Create 200x200 Canvas widget
my_canvas2.place(x=550,y=420)

my_oval2 = my_canvas2.create_oval(5, 5, 50, 50,width=4)  # Create a circle on the Canvas
#my_oval1 = my_canvas.create_oval(5, 170, 150, 300,width=7)
outputText="READY"
var = StringVar()
l = Label(root, textvariable=var,wraplength=398,font=("times new roman",28,"bold"),bg="WHITE")
l.place(x=680,y=350)
var.set("")






Label(root,text="CAPTURED",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=330,y=220)
f2=LabelFrame(root,bg="black")
f2.place(x=340,y=10)
live1=Label(f2,bg="black")
live1.pack()

Label(root,text="TEMPLATE",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=630,y=220)
f3=LabelFrame(root,bg="black")
f3.place(x=620,y=10)
live2=Label(f3,bg="black")
live2.pack()

Label(root,text="RESULT",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=680,y=750)
f4=LabelFrame(root,bg="black")
f4.place(x=200,y=250)
live3=Label(f4,bg="black")
live3.pack()

Label(root,text="Match_percentage",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=10,y=260)
f5=LabelFrame(root,bg="black")
f5.place(x=10,y=270)

#inputtxt = Entry(root)
#inputtxt.place(x=10,y=600)
match_percent=StringVar(root)
match_percent.set("80")
sp = Spinbox(root, from_= 0, to = 100,textvariable=match_percent)
sp.place(x=10,y=280)
global temp0
global temp1
global temp2
def find_match(template_name):
    img_rgb = cv2.imread('saved_img.jpg')
    template = cv2.imread(template_name, 1)
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
        
        if template_name=="crop0.jpg":
            my_canvas.itemconfig(my_oval, fill="GREEN")
            var.set("Temp1 OK")
            temp0=True
            print(temp0)
        else:
            start_point = (230,53)
            end_point = (139,137)
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
            my_canvas.itemconfig(my_oval, fill="RED")
            var.set("Temp1 NOT OK")

        if template_name=="crop1.jpg":
            my_canvas1.itemconfig(my_oval1, fill="GREEN")
            var.set("Temp2 OK")
            print(temp1)

        else:
            start_point = (230,53)
            end_point = (139,137)
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
            my_canvas1.itemconfig(my_oval1, fill="RED")
            var.set("Temp2 NOT OK")
        #GPIO.output(NOT_OK_signal, GPIO.HIGH)

        if template_name=="crop2.jpg":
            my_canvas2.itemconfig(my_oval2, fill="GREEN")
            var.set("Temp3 OK")
            temp2=True
            print(temp2)
        else:
            start_point = (230,53)
            end_point = (139,137)
            color = (0, 0, 255)
            thickness = 4
            cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
            my_canvas2.itemconfig(my_oval2, fill="RED")
            var.set("Temp3 NOT OK")

        #GPIO.output(OK_signal, GPIO.HIGH)
        time.sleep(2)
        #GPIO.output(OK_signal, GPIO.LOW)
        print ((pt[0] + w, pt[1] + h))
        break

# Show the final image with the matched area.

        #GPIO.output(NOT_OK_signal, GPIO.HIGH)
    

    time.sleep(2)
        #
        # 
        #GPIO.output(NOT_OK_signal, GPIO.LOW)
    #cv2.imshow('Detected', img_rgb)
    #cv2.waitKey(0)
    cv2.imwrite(filename='Result.jpg', img=img_rgb)

def findmatch1():
    find_match("crop0.jpg")
    find_match("crop1.jpg")
    find_match("crop2.jpg")

def template_capture():
    img_raw = cv2.imread("saved_img.jpg")

    bcvalue=open("bcvalues.txt","w") 
    bcvalue.write(str(brightness_value)+"  "+str(contrast_value))
    bcvalue.close()
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

b1=Button(root,text="Capture",font=("Arial Black",10,"bold"))
b1.config(command=capture_image)
b1.place(x=500,y=220)

b2=Button(root,text="Match",font=("Arial Black",10,"bold"))
b2.config(command=findmatch1)
b2.place(x=10,y=320)

b3=Button(root,text="Template_capture",font=("Arial Black",10,"bold"))
b3.config(command=template_capture)
b3.place(x=800,y=220)


def controller(img, brightness, contrast):
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

w1=Scale(root,from_=0,to=255,orient=VERTICAL,cursor="circle")
w1.set(brightness_value)
w1.place(x=0,y=110)
w2=Scale(root,from_=0,to=255,orient=VERTICAL,cursor="circle")
w2.set(contrast_value)
w2.place(x=0,y=0)



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
    img=cv2.resize(img,(250,200))
    img1=Image.fromarray(img)
    img1=ImageTk.PhotoImage(img1)
    live["image"]=img1
    capture_img=cv2.imread("saved_img.jpg")
    capture_img=cv2.cvtColor(capture_img,cv2.COLOR_BGR2RGB)
    capture_img=cv2.resize(capture_img,(250,200))
    capture_img1=Image.fromarray(capture_img)
    capture_img2=ImageTk.PhotoImage(capture_img1)
    live1["image"]=capture_img2

    template_img=cv2.imread("saved_img1.jpg")
    template_img=cv2.cvtColor(template_img,cv2.COLOR_BGR2RGB)
    template_img=cv2.resize(template_img,(250,200))
    template_img1=Image.fromarray(template_img)
    template_img2=ImageTk.PhotoImage(template_img1)
    live2["image"]=template_img2

    result_img=cv2.imread("Result.jpg")
    result_img=cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)
    result_img=cv2.resize(result_img,(300,250))
    result_img1=Image.fromarray(result_img,mode=None)
    result_img2=ImageTk.PhotoImage(result_img1)
    live3["image"]=result_img2
    root.update()
    #print(GPIO.input(Trigger))
    #if GPIO.input(Trigger)== 1:
       # capture_image()
       # cv2.waitKey(500)
        #find_match()

    