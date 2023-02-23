
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import sys
import time 

# supressing runtime warnings 
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)


lower = (0, 165, 112)
upper = (180, 255, 255)

#defining path of file and object for videocapture 
path = r".\source\ball.mov"
video = cv2.VideoCapture(path)
time.sleep(2.0)



#defining list for x and y points
x_points=[]
y_points=[]



def least_squares(x_points,y_points):
	x_matrix=np.vstack((x_points**2,x_points,np.ones(x_points.shape))).transpose()
	print(x_matrix.shape)
	x_transpose_x=np.linalg.inv(np.matmul(x_matrix.transpose(),x_matrix))
	x_y=np.matmul(x_matrix.transpose(),y_points)
	A=np.matmul(x_transpose_x,x_y)
	return A


while True:

	frame = video.read()
	frame = frame[1] 
	if frame is None:
		break
	
	# cv2.imshow("frame without mask",frame)
	frame1=frame
	# print(frame.shape)
	cv2.rectangle(frame,(0,450),(600,600),(0,0,0),-1)
	cv2.imshow("frame",frame)

	#noise reduction and blurring
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	
	#masking 
	mask = cv2.inRange(hsv, lower, upper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	cv2.imshow("mask",mask)

	#using matplotlib to verify coordinates of x and y 
	# plt.imshow(frame)
	# plt.show()
	# plt.imshow(mask)
	# plt.show()

	#filtering out pixel indexes with 255 
	pixels = np.where(mask == [255])
	#ensuring pixels have a value 
	if np.all(pixels):
		x=np.mean(pixels[1])
		y=np.mean(pixels[0])
		# print(x,y)
	
	#ignoring frames without ball in them
	if not np.isnan(x) and not np.isnan(y):
		x_points.append(x)
		y_points.append(y)
	
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
			break

	#for frame by frame checking
	# while key not in [ord('q'), ord('k')]:
	# 	key = cv2.waitKey(0)
	# 	if key == ord("q"):
	# 		break


cv2.destroyAllWindows()

#converting list to numpy array
shapeofimage=frame1.shape
y_points=np.array(y_points)
x_points=np.array(x_points)

#least squares and calculating y after fitting 
a=least_squares(x_points,y_points)
print("The equation of parabola is ","%fx^2"%a[0],"%fx+"%a[1],"%f = 0"%a[2])
calculated_y=(a[0]*np.square(x_points))+(a[1]*x_points)+a[2]

# finding X value given Y value 
y_1=calculated_y[0]+300
b_4ac= (a[1]**2)-(4*a[0]*(a[2]-y_1))
a_b_1=-a[1]-np.sqrt(b_4ac)
a_b_2=-a[1]+np.sqrt(b_4ac)
x_1=a_b_1/(2*a[0])
x_2=a_b_2/(2*a[0])
print("The approx pixel where the ball would land is",x_2)

#plotting, changing x and y poitns from top left origin to bottom left of the plot 
fig,ax=plt.subplots()
# plt.imshow(frame1)
# plt.scatter(x_points,y_points)
plt.scatter(x_points,(shapeofimage[0]-y_points),linewidths=.1,marker='.',label="Original Path")
plt.scatter(x_points,(shapeofimage[0]-calculated_y),marker='.',linewidths=.1,label="Fitted Path")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Plot of Path of Ball and Fitted Equation of Parabola ")
plt.legend()
plt.show()



