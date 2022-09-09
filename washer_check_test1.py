from ast import Break, Continue
from cProfile import label
from cgitb import text
from copyreg import pickle
from ctypes import resize
from email.mime import image
from operator import index
import string
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
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


#GPIO.setmode(GPIO.BCM)
#GPIO.setup(OK_signal, GPIO.OUT)
#GPIO.setup(Trigger, GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#GPIO.setup(NOT_OK_signal, GPIO.OUT)




bc_value_retain = open("bcvalues.txt","r") 
#print(bc_value_retain.read(5))
bc_value_string=bc_value_retain.read()
#print(bc_value_string)
brightness_value=bc_value_string[0:3]
contrast_value=bc_value_string[3:9]
print(brightness_value)
print(contrast_value)
bc_value_retain.close()
brightness_value=int(brightness_value)
contrast_value=int(contrast_value)
key = cv2. waitKey(1)
cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,200)
cap.set(cv2.CAP_PROP_EXPOSURE,-3)
cap.set(cv2.CAP_PROP_FPS,6)
cap.set(cv2.CAP_PROP_ZOOM,0)
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

my_canvas =Canvas(root, width=120, height=120,background="white")  # Create 200x200 Canvas widget
my_canvas.place(x=550,y=290)

my_oval = my_canvas.create_oval(5, 5, 120, 120,width=7)  # Create a circle on the Canvas
#my_oval1 = my_canvas.create_oval(5, 170, 150, 300,width=7)
outputText="READY"
var = StringVar()
l = Label(root, textvariable=var,wraplength=398,font=("times new roman",30,"bold"),bg="WHITE")
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

Label(root,text="RESULT",bg="white",fg="black",font=("times new roman",12,"bold")).place(x=680,y=250)
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


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def conture_Detect_Measure():
    # read the image
    image = cv2.imread('washer_color_extracted.jpg')
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    #cv2.imshow("dilated",edged)
    #edged = cv2.erode(edged, None, iterations=2)
    #cv2.imshow("erode",edged)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    #print("number of cnts: ")
    #print(type(cnts))
    #print(cnts)
    if cnts[1] is not None:
    #if cnts[1].any!= None:
        cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        for c in cnts:
            if cv2.contourArea(c) < 20:
                my_canvas.itemconfig(my_oval, fill="ORANGE")
                capture_image()
                conture_Detect_Measure()
                cv2.imwrite(filename='saved_img.jpg', img=orig)
            else:
                orig = cv2.imread("saved_img.jpg")
                print(cv2.contourArea(c))
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                box=box+140
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                    (tl, tr, br, bl) = box
                    (tltrX, tltrY) = midpoint(tl, tr)
                    (blbrX, blbrY) = midpoint(bl, br)
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)
                    #cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                    #cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                    #cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                    #cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                    #cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
                    #cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
                    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    if pixelsPerMetric is None:
                        pixelsPerMetric = dB / 1.0
                    dimA = dA / pixelsPerMetric
                    #print(dA)
                    #dA=dA*100
                    dA=int(dA)
                    print(dA)
                    dimB = dB / pixelsPerMetric
                    print(int(dB))
            #range1=range(0.15,0.25)
                    okrange=np.arange(0.15,0.25,0.1)
                    nokrange=np.arange(0.24,0.35,0.1)
                    if 5<dA<12:
                        my_canvas.itemconfig(my_oval, fill="GREEN")
                        var.set("OK")
                        print("job is ok")
                    elif 12<dA<40:
                        my_canvas.itemconfig(my_oval, fill="RED")
                        var.set("2 washer nok")

                    cv2.putText(orig, "{:.1f}in".format(dA),
		            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                    #cv2.putText(orig, "{:.1f}in".format(dimB),
		            #(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
                    #
                    #cv2.imshow("Image", orig)
                    cv2.imwrite(filename='Result.jpg', img=orig)
                    #cv2.waitKey(0)
    else:
        print("cnts not found")
        my_canvas.itemconfig(my_oval, fill="RED")
        var.set("No washer nok")
        saved_img=cv2.imread("saved_img.jpg")
        cv2.imwrite(filename='Result.jpg', img=saved_img)
    #cv2.drawContours(image=orig, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
# see the results
    #cv2.imshow('None approximation', orig)


def find_match():
    img_rgb = cv2.imread('roi1.jpg')
    #x1=124
    #x2=169
    #y1=136
    #y2=172
    #img_rgb = img_rgb[y1:y1+y2,x1:x1+x2]
    template = cv2.imread('saved_img1.jpg', 1)
    #cv2.imshow("roi",img_rgb)
    #cv2.imwrite(filename='roi1.jpg', img=img_rgb)
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
    print(np.where(res))
    #print(str(loc))
# Draw a rectangle around the matched region.
    flag=False
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0),7 )
        #print(pt)
        #print((pt[0] + w, pt[1] + h))
        flag=True
        my_canvas.itemconfig(my_oval, fill="GREEN")
        var.set("OK")
        #GPIO.output(OK_signal, GPIO.HIGH)
        time.sleep(2)
        #GPIO.output(OK_signal, GPIO.LOW)
        print ((pt[0] + w, pt[1] + h))
        break

# Show the final image with the matched area.

    if flag==False:
        start_point = (230,53)
        end_point = (139,137)
        color = (0, 0, 255)
        thickness = 4
        cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
        my_canvas.itemconfig(my_oval, fill="RED")
        var.set("NOT OK")
        #GPIO.output(NOT_OK_signal, GPIO.HIGH)
        time.sleep(2)
        #GPIO.output(NOT_OK_signal, GPIO.LOW)
    #cv2.imshow('Detected', img_rgb)
    #cv2.waitKey(0)
    cv2.imwrite(filename='Result.jpg', img=img_rgb)


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

    bcvalue=open("roi_points_retain.txt","w") 
    bcvalue.write(str(int(roi[1])).zfill(3)+"  "+str(int(roi[1]+roi[3])).zfill(3)+" "+str(int(roi[0])).zfill(3)+" "+str(int(roi[0]+roi[2])).zfill(3))
    bcvalue.close()

    cv2.imwrite(filename='saved_img1.jpg', img=roi_cropped)

    cv2.destroyAllWindows()


def capture_image():

    roi_points=open("roi_points_retain.txt","r")

    roi_points_retain=roi_points.read()

    roi_y1=int(roi_points_retain[0:3])
    roi_y2=int(roi_points_retain[5:8])
    roi_x1=int(roi_points_retain[9:12])
    roi_x2=int(roi_points_retain[13:16])

    print(roi_x1,roi_y1,roi_x2,roi_y2)
    image=Image.fromarray(img2)
    image.save("saved_img.jpg")
    img_rgb_roi = cv2.imread('saved_img.jpg')
    x1=roi_x1
    y1=roi_y1
    x2=roi_x2
    y2=roi_y2
    img_rgb_roi = img_rgb_roi[y1:y2,x1:x2]
    #img_rgb_roi = img_rgb_roi[y1:y1+y2,x1:x1+x2]
    cv2.imwrite(filename='roi1.jpg', img=img_rgb_roi)
    img_rgb_roi = cv2.imread('roi1.jpg')
    hsv = cv2.cvtColor(img_rgb_roi, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV 
    lower_blue = np.array([3,0,0])
    upper_blue = np.array([18,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image

    res = cv2.bitwise_and(img_rgb_roi,img_rgb_roi, mask= mask)
    cv2.imwrite(filename='washer_color_extracted.jpg', img=res)
    #cv2.imshow('frame',img_rgb_roi)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
   

b1=Button(root,text="Capture",font=("Arial Black",10,"bold"))
b1.config(command=capture_image)
b1.place(x=500,y=220)

b2=Button(root,text="Match",font=("Arial Black",10,"bold"))
b2.config(command=conture_Detect_Measure)
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

w1=Scale(root,from_=-10,to=+10,orient=VERTICAL,cursor="circle")
w1.set(brightness_value)
w1.place(x=0,y=110)
w2=Scale(root,from_=0,to=10000,orient=VERTICAL,cursor="circle")
w2.set(contrast_value)
w2.place(x=0,y=0)



while True:
    

    brightness_value=int(w1.get())
    contrast_value=int(w2.get())

    # kernal=slider3.get()
    # rang=slider4.get()
    img=cap.read()[1]
    #img=controller(img,brightness_value,contrast_value)
    #img3=cv2.detailEnhance(img,255,0.1)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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

    '''
    if GPIO.input(Trigger)== 1:
        capture_image()
        cv2.waitKey(500)
        find_match()'''
    