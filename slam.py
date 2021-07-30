#!/usr/bin/env python3
import cv2
import time
import numpy as np
from display import Display 
from extractor import Extractor 

H = 1280//4
W = 600//1

disp = Display(W, H)
fe = Extractor()

def process_frame(img):
	img = cv2.resize(img, (W,H))
	matches = fe.extract(img)


	for pt1, pt2 in matches:
		u1,v1 = map(lambda x: int(round(x)), pt1.pt)
		u2,v2 = map(lambda x: int(round(x)), pt2.pt)
		cv2.circle(img, (u1,v1), color=(0,255,0), radius=2)
		cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))
		#cv2.rectangle(img, (u1,v1), (u2,v2), color=(255,0,0))

	disp.point(img)

if __name__ == "__main__":
	#cap = cv2.VideoCapture(cv2.CAP_V4L2)
	cap = cv2.VideoCapture('yolo_test.mp4')

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			continue

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break