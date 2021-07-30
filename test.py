import urllib.request
import time
import numpy as np 
import cv2
from display import Display
from extractor import Extractor

url='http://10.1.219.218:8080/shot.jpg'
#url='http://192.168.43.10:8080/shot.jpg'
H = 1280//4
W = 600//1

disp = Display(W, H)
fe = Extractor()

#time.sleep(0.1)
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
	imgResp = urllib.request.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	image = cv2.imdecode(imgNp,-1)
	#print(imgNp)

	# for object detection
	'''
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	boxes, weights = hog.detectMultiScale(image, winStride=(8,8) )
	boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

	for (xA, yA, xB, yB) in boxes:
		cv2.rectangle(image, (xA, yA), (xB, yB),(0, 255, 0), 2)
		cv2.circle(image, (xA,yA), color=(0,255,0), radius=2)
	'''
#-----------------------------------------------------------------------
	img = cv2.resize(image, (W,H))
	matches = fe.extract(img)


	for pt1, pt2 in matches:
		u1,v1 = map(lambda x: int(round(x)), pt1.pt)
		u2,v2 = map(lambda x: int(round(x)), pt2.pt)
		cv2.circle(img, (u1,v1), color=(0,255,0), radius=2)
		cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))
		#cv2.rectangle(img, (u1,v1), (u2,v2), color=(255,0,0))

	disp.point(img)

	#time.sleep(0.1)

	img = cv2.resize(image,(1366//2,766//2))
	#cv2.imshow("Frame", img);

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break	






