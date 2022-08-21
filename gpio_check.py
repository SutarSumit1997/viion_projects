import RPi.GPIO as GPIO
import time

led = 21
led1 = 20
switch = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(led, GPIO.OUT)
GPIO.setup(switch, GPIO.IN)
GPIO.setup(led1, GPIO.OUT)

for i in range(100):
    GPIO.output(led, GPIO.LOW)
    time.sleep(1)

    GPIO.output(led, GPIO.HIGH)
    time.sleep(10)
    print('Switch status = ', GPIO.input(switch))
GPIO.output(led, GPIO.LOW)
GPIO.output(led1, GPIO.LOW)
GPIO.cleanup()

