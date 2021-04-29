# import the necessary packages
import numpy as np
import argparse
import math
from math import cos
from math import sin
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import json
from PIL import Image
from sympy import sympify
from sympy.geometry import Point2D, Segment2D, Circle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sqrt
import re
import os
from collections import defaultdict

def intersec(x,y,b1,b2,c1,c2,r):
	A = Point2D(x,y)
	B = Point2D(b1,b2) #line1
	C = Point2D(c1,c2) #line2

	# Segment from A to B
	f_0 = Segment2D(A, B)
	# Segment from A to C
	f_1 = Segment2D(A, C)
	# Circle with center A and radius 118 
	c = Circle(A, sympify(r, rational=True))


	i_0 = c.intersection(f_0)
	i_1 = c.intersection(f_1)
	# print(i_0)
	# print(i_1)
	#\((.*)

	#ก้อน3 \,(.*)\)\]
	# re.split(r',\s*(?![^()]*\))', i_0)
	#จุดที่ 1
	string1 = re.split('\((.*)', str(i_0))
	string2 = re.split('(.*)\,', string1[1])
	string3 = re.split('\,(.*)\)\]', string1[1])

	#จุดที่2
	string4 = re.split('\((.*)', str(i_1))
	string5 = re.split('(.*)\,', string4[1])
	string6 = re.split('\,(.*)\)\]', string4[1])

	outputstr1 = string2[1]
	outputstr2 = string3[1]

	outputstr3 = string5[1]
	outputstr4 = string6[1]

	output1 = eval(outputstr1)
	output2 = eval(outputstr2) 

	output3 = eval(outputstr3)
	output4 = eval(outputstr4) 

	p1 = int(output1)
	p2 = int(output2)

	p3 = int(output3)
	p4 = int(output4)

	return p1,p2,p3,p4

def quadrant(x, y,x1,y1):
	if (x > x1 and y > y1):
		# print ("lies in Fourth quadrant")
		q=4
	
	elif (x < x1 and y > y1):
		# print ("lies in Third quadrant")
		q=3
    
	elif (x < x1 and y < y1):
		# print("lies in Second quadrant")
		q=2 
	
	elif (x > x1 and y < y1): 
		# print ("lies in First quadrant")
		q=1

	else:
		# print ("lies at origin") 
		q=0.14

	return q


#data[0][0] = ymin , data[0][1]=ymax, data[0][2]=xmin, data[0][3]=xmax

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

def draw(x1,y1,r,ang):
	angle = ang+15
	length = r
	θ = angle * 3.14 / 180
	x2 = x1 + length * cos(θ)
	y2 = y1 + length * sin(θ) 
	return x2,y2

def draw_1(x1,y1,r,angle):
    ang = angle * 3.14 / 180
    x2 = x1 + r * cos(ang)
    y2 = y1 + r * sin(ang) 
    return x2,y2

def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m
	
def checkInArea(top_left,bottom_right,p):
    if (p[0] > top_left[0] and p[0] < bottom_right[0] and p[1] > top_left[1] and p[1] < bottom_right[1]) :
        return True
    else :
        return False

# check class number
def checkNumberClass(text):
    if(text == "one"):
        return 1
    elif(text == "two"):
        return 2
    elif(text == "three"):
        return 3
    elif(text == "four"):
        return 4
    elif(text == "five"):
        return 5
    elif(text == "six"):
        return 6
    elif(text == "seven"):
        return 7
    elif(text == "eight"):
        return 8
    elif(text == "nine"):
        return 9
    elif(text == "ten"):
        return 10
    elif(text == "eleven"):
        return 11
    elif(text == "twelve"):
        return 12
#  check line in number's area
def inArea(p,data) :
    number = 0
    status = False
    for i in data:
        top_left = (int(i[2]), int(i[0]))
        bottom_right = (int(i[3]), int(i[1]))
        text = str(i[5][0]).split(':')
        # print(top_left,bottom_right,p,text[0])
        check = checkInArea(top_left,bottom_right,p)
        if(check == True):
            status = True
            num_text = str(i[5][0]).split(':')
            # print("true",num_text[0])
            number = checkNumberClass(num_text[0])
    return status,number
def check_clockwise(image,data,x_center,y_center,radius):
	# print("data clock", len(data))
	output = image.copy()
	line_list=[]
	for i in data:
		cv2.rectangle(output, (i[2], i[0]),(i[3], i[1]), (76, 179, 111), 5)
		cv2.putText(output,str(i[5]),(i[3], i[1]), font, 0.8, (76, 179, 111), 2, cv2.LINE_AA)

	x1,y1 = x_center,y_center
	an = 0
	print(x_center,y_center,radius)
	number_list = []
	for i in range(0,60):
	# check each line
		x2,y2 = draw(x1,y1,radius,an)
		#cv2.line(output,(x1,y1),(int(x2),int(y2)),(0,0,0),(1))
		getSlope = slope(x1,y1,x2,y2)
		c1 = y1-(getSlope*x1)
		a,b = x1,x2
		c,d = y1,y2
		if (x1 < x2):
			a = x1
			b = x2
		elif (x1 > x2):
			a = x2
			b = x1
		if (y1 < y2):
			c = y1
			d = y2
		elif (y1 > y2):
			c = y2
			d = y1
		
		num_in_line = []
	# check each point 90 and 270
		if an == 90 or an == 270:
			for i in range(int(c),int(d)):
				newX = (i - c1)/getSlope
				status,num = inArea((int(newX),i),data)
				# cv2.circle(output, (int(newX),i),2, (0, 128, 255), 2)
				if (status):
					if (len(num_in_line) == 0):
						num_in_line.append(num)
					else:
						if (num != num_in_line[len(num_in_line)-1]):
							num_in_line.append(num)
		else:
			for i in range(int(a),int(b)):
				newY = (getSlope*i)+ c1
				status,num = inArea((i,int(newY)),data)
				# cv2.circle(output, (i,int(newY)),2, (0, 128, 255), 2)
				if (status):
					if (len(num_in_line) == 0):
						num_in_line.append(num)
					else:
						if (num != num_in_line[len(num_in_line)-1]):
							num_in_line.append(num)
		if len(num_in_line) != 0:
			number_list.append(num_in_line)
		an = an+6
	#   print(number_list)
	size = len(number_list)
	status = 0
	# 0 is null, 1 is clockwise , -1 is anti-clockwise 
	pv_total = 0
	total = "-"
	over = 0 
	count = 0
	# print("size",size)
	for i in range(0,size):
		if (i != size-1):
			# print(len(number_list[i]))
			if len(number_list[i]) == 1:
				if number_list[i] != number_list[i+1]:
					diff = number_list[i][0]-number_list[i+1][0]
					if total == "-": 
						total = diff
						pv_total = total
					else:
						total = total+diff
					if total < pv_total:
						if status == 0:
							status = 1
						elif status != 1:
							over = over + 1
					elif total > pv_total:
						if status == 0:
							status = -1
						elif status != -1:
							over = over + 1
					# print(number_list[i][0],"-",number_list[i+1][0],total,pv_total)
					# print(status)
					pv_total = total
				elif len(number_list[i]) != 1:            
					count = count +1
				if count > 1:
					status = 2
					break
	# print(status)
	if over == 1:
		if status == 1:
			mes = "clockwise"
			print("clockwise")
		elif status == -1:
			mes = "anti-clockwise"
			print("anti-clockwise")
	else:
		mes = "other form arrange"
		print("other form arrange")
	return output,mes

def checkpoint(list,listp,classnum):
	# cv2.circle()
	print("list of boxnum : ",list[classnum-1][:4])
	print("list tee kumlung check num in box:",list[classnum-1])
	p1,p2 = listp
	point = Point(p1, p2)
	polygon = Polygon(list[classnum-1][:4])
	#if(polygon.contains(point) == True):
	#while(True):
	#checkInArea(top_left,bottom_right,p)
	#[(399, 428), (381, 361), (434, 330), (483, 379)]
	
	# print(polygon.contains(point))
	# print(polygon.contains(point))

	return polygon.contains(point),classnum

def changestrtoint(c1,c2,name):
	if (name=='one'):
		name=1
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='two'):
		name=2
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='three'):
		name=3
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='four'):
		name=4
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='five'):
		name=5
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='six'):
		name=6
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='seven'):
		name=7
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='eight'):
		name=8
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='nine'):
		name=9
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='ten'):
		name=10
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='eleven'):
		name=11
		c = c1,c2,name
		# print(sort_list_centroid)
	elif (name=='twelve'):
		name=12
		c = c1,c2,name
		# print(sort_list_centroid)
	return c,int(name)


#ymin,ymax,xmin,xmax bounding box 
def check_quardrant(name,total,xmin):
	global c_errorQ1,c_errorQ2,c_errorQ3,c_errorQ4
	if (total==4):
		mes = name+" in quardrant 1"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="one" and name!="two"):
			c_errorQ1 = c_errorQ1 + 1
	elif (total==8):
		mes = name+" in quardrant 2"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="ten" and name!="eleven"):
			c_errorQ2 = c_errorQ2 + 1
		print(c_errorQ2)
	elif (total==12):
		mes = name+" in quardrant 3"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="seven" and name!="eight"):
			c_errorQ3 = c_errorQ3 + 1
	elif (total==16):
		mes = name+" in quardrant 4"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="four" and name!="five"): 
			c_errorQ4 = c_errorQ4 + 1
	elif (total==6):
		mes =name+ " between q1 & q2"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
	elif (total==10):
		if(xmin<x):
			mes =name+ " between between q2 & q3"
			_,n = changestrtoint(0,0,name)
			l = mes,name,n
			list_ofq.append(l)
		elif(xmin>=x):
			mes =name+ " between q1 & q4"
			_,n = changestrtoint(0,0,name)
			l = mes,name,n
			list_ofq.append(l)
			
	elif (total==14):
		mes =name+ " between q3 & q4"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		
	else:
		pc = quadrant((xmin+xmax)/2, (ymin+ymax)/2,x,y)
		check_center(name,pc,list_ofq)
		

def check_center(name,num,list_ofq):
	global c_errorQ1,c_errorQ2,c_errorQ3,c_errorQ4
	if (num==1):
		mes = name+" in quardrant 1"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="one" and name!="two"):
			c_errorQ1 = c_errorQ1 + 1
		print(c_errorQ1)
	elif (num==2):
		mes = name+" in quardrant 2"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="ten" and name!="eleven"):
			c_errorQ2 = c_errorQ2 + 1
	elif (num==3):
		mes = name+" in quardrant 3"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="seven" and name!="eight"):
			c_errorQ3 = c_errorQ3 + 1
		print(c_errorQ3)
	elif (num==4):
		mes = name+" in quardrant 4"
		_,n = changestrtoint(0,0,name)
		l = mes,name,n
		list_ofq.append(l)
		if(name!="four" and name!="five"):
			c_errorQ4 = c_errorQ4 + 1

# Image.fromarray(output).show()
def checklist(temp):
	ordered_list = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve']
	num_list = ['1','2','3','4','5','6','7','8','9','10','11','12']
	# print(ordered_list)
	#print(temp)
	list_ = [x for x in ordered_list if x not in temp]
	com = set(temp) == set(ordered_list)
	return com,list_ 

def changeclass(dif):
    arraylist = []
    my_array = np.array(dif)
    print (len(my_array))
    # print (my_array[0])
    for i in range(len(my_array)):
        if (my_array[i] == 'one'):
            arraylist.append(1)
        elif (my_array[i] == 'two'):
            arraylist.append(2)
        elif (my_array[i] == 'three'):
            arraylist.append(3)
        elif (my_array[i] == 'four'):
            arraylist.append(4)
        elif (my_array[i] == 'five'):
            arraylist.append(5)
        elif (my_array[i] == 'six'):
            arraylist.append(6)
        elif (my_array[i] == 'seven'):
            arraylist.append(7)
        elif (my_array[i] == 'eight'):
            arraylist.append(8)
        elif (my_array[i] == 'nine'):
            arraylist.append(9)
        elif (my_array[i] == 'ten'):
            arraylist.append(10)
        elif (my_array[i] == 'eleven'):
            arraylist.append(11)
        elif (my_array[i] == 'twelve'):
            arraylist.append(12)
    # print("class-diff: ",arraylist)

# Name of the directory containing the object detection module we're using
#IMAGE_NAME = '2aisvtplb'
RES_FILE = '_scorenum.jpg'
FILE = '_num.jpg'
DETECT_FOLDER = 'detectcircle'
IMAGE_FOLDER = 'image_test'
RESULT_FOLDER = 'result\CDT_rewrite'
ALL_RES = 'result\\new_score'
CDT_REWRITE = 'image_test\CDT_rewrite'
# IMAGE_WITHPREDICT = 'result\CDT_rewrite'
IMAGE_WITHPREDICT = 'result\\new'
IMAGETEST_FOLDER = 'image_test'

# Grab path to current working directory
CWD_PATH = os.getcwd()
PREVIOS_PATH = os.path.abspath(CWD_PATH+ "/../")
folder = os.path.join(CWD_PATH,IMAGETEST_FOLDER,'CDT_rewrite')
id_folder = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
font=cv2.FONT_ITALIC

for i in range(0,len(id_folder)):
	try:
		#print(i,id_folder[i])
		#IMAGE_NAME = str(id_folder[i])
		IMAGE_NAME='test11'
		line_list = []
		listofp = [] 
		line=[]
		name=[]
		listname=[]
		list_digit=[]
		list_centroid =[]
		list_ofq =[]
		sort_list_centroid = []
		boxofnum =[]
		arraylist=[]
		list_boolean =[]
		c_errorQ1 = 0 
		c_errorQ2 = 0 
		c_errorQ3 = 0 
		c_errorQ4 = 0 
		# Path to image
		PATH_TO_IMAGE = os.path.join(IMAGE_WITHPREDICT,IMAGE_NAME+'.png')
		#print(PATH_TO_IMAGE)

		#path to save result
		PATH_TO_RESULT = os.path.join(RESULT_FOLDER,IMAGE_NAME+RES_FILE)
		PATH_TO_ALLRESULT = os.path.join(ALL_RES,IMAGE_NAME+RES_FILE)
		#print(PATH_TO_RESULT)
		#load json file
		new_data_corr = []
		data_circle = []
		with open("json_num/CDT_rewrite/script_"+IMAGE_NAME+".json") as f:
			data = json.load(f)
		for p in data['coordinate']:
			new_data_corr.append(p)
		for p in data['circle']:
			data_circle.append(p)
		print("data_corr:",new_data_corr)
		x,y,r = data_circle
		image = cv2.imread(PATH_TO_IMAGE)
		output = image.copy()
		num = r*(40/100)
		in_r = r-int(num)
		cv2.circle(output,(x, y),in_r, (255,0,0), 3)
		j=2 #เส้นแรกอยู่บนเลข 5 ละวนตามเข็ม
		for i in range(30,385,30):
			x2,y2 = draw(x,y,r,i)
			xend = int(x2)
			yend = int(y2) 
			cv2.line(output,(x,y),(xend,yend),(224, 45, 45),2)
			line_list.append((xend,yend))

		#print("line_list: ",line_list)
		length = len(line_list)
		for b in range(length):
			if (b==11):
				p1,p2,p3,p4 = intersec(x,y,*(line_list[b]),*(line_list[0]),in_r)
				pointcut = [line_list[0],(p3,p4),(p1,p2),line_list[b]]
				listofp.append(pointcut)

			else:
				p1,p2,p3,p4 = intersec(x,y,*(line_list[b]),*(line_list[b+1]),in_r)
				pointcut = [line_list[b+1],(p3,p4),(p1,p2),line_list[b]]
				listofp.append(pointcut)

		#load coordinates from json file
		for i in range(0, len(new_data_corr)):
			line.append(new_data_corr[i])
			ymin  = line[i][0]
			ymax = line[i][1]
			xmin = line[i][2]
			xmax = line[i][3]
			name = line[i][5][0].split(":")
			#print(name)
			listname.append(str(name[0]))
			p1 = quadrant(xmin,ymin,x,y)
			p2 = quadrant(xmax,ymin,x,y)
			p3 = quadrant(xmin,ymax,x,y)
			p4 = quadrant(xmax,ymax,x,y)
			total = p1+p2+p3+p4
			j, k = (xmin+xmax)/2, (ymin+ymax)/2
			list_centroid.append((int(j),int(k),str(name[0])))
			# Draw a circle in the center of rectangle
			cv2.circle(output, center=(int(j), int(k)), radius=3, color=(255, 0, 0), thickness=5)
			check_quardrant(name[0],total,xmin)

		match,dif = checklist(listname)
		#print("match? : ",match)
		#print("diff= " ,dif) #คลาสที่ไม่มี 
		for i in listname:
			print(i)
			digit = checkNumberClass(i)
			list_digit.append(digit)


		list_digit.sort()
		#print("list_digit:",list_digit)

		total_point = 0
		score_1 = 0
		score_2 = 0
		score_3 = 0
		point1 = 0
		point2 = 0
		point3 = 0
		changeclass(dif)
		#print("list of quardrant: ",list_ofq)
		list_ofq.sort(key = lambda x: x[2])  

		# print("sort: ",list_ofq)
		for i in range(0, len(list_ofq)):
			print(list_ofq[i][0])

		j = 5
		L=0
		for i in list_centroid:
			c1,c2,name = i
			c,_ = changestrtoint(c1,c2,name)
			sort_list_centroid.append(c)

		sort_list_centroid.sort(key = lambda x: x[2])  
		#print("sort_list_centroid(1): ",sort_list_centroid)
		#start intercept point5
		Y = [5,6,7,8,9,10,11,12,1,2,3,4]
		sort_listofp = [listofp for _,listofp in sorted(zip(Y,listofp))]
		# add class in sort_listofp 
		for k in range(0, len(sort_listofp)):
			result = [[m, n, s,t,k+1] for m, n, s,t in sort_listofp]
			boxofnum.append(result[k])

		#print("boxofnum:",boxofnum)
		#print("sort_list_centroid(2)",sort_list_centroid)
		cen=[]
		item=[]
		if(len(boxofnum)>=len(sort_list_centroid)):
			size = len(boxofnum)
		if(len(sort_list_centroid)>len(boxofnum)):
			size = len(sort_list_centroid)

		for j in range (len(sort_list_centroid)):
			#print("j:",j)
			a,b = sort_list_centroid[j][:2]
			#print("num:",sort_list_centroid[j][2])
			#print("centriod:",sort_list_centroid[j][:2])
			boolean,classnum = checkpoint(boxofnum,sort_list_centroid[j][:2],sort_list_centroid[j][2])
				# print(boolean)
			try:	
				list_boolean.append(boolean)
				cv2.putText(output,str(boolean), (int(a),int(b)), font, 1, (116, 7, 97), 2, cv2.LINE_AA)
			except OSError as error:
				print(error)  
				Image.fromarray(output).show()
		print("list_boolean:",list_boolean)
		
		if False in list_boolean :
			point3 = 0
		if len(list_boolean) == 0:
			point3 = 0
		else:
			point3 = 1

		#1.clockwise 2.arrange  3.located 1:yes 0:no
		#clockwise
		output,mes = check_clockwise(output,new_data_corr,x,y,r)
		if(mes=="clockwise"):
			point1 = 1
			point2 = 1
		elif(mes=="anti-clockwise"):
			point1 = 0
			point2 = 1
		elif(mes=="other form arrange"):
			point1 = 0
			point2 = 0
		else: 
			point1 = 0
			point2 = 0


		total_point = point1+point2+point3
		if (total_point==3):
			score_3 = 2
		elif (total_point==0):
			score_3 = 0
		else:
			score_3 = 1

		#test
		#dif = ['1','2','3','6']
		#score rule-1(1-12 on clock) 
		if(len(dif)<=1):
			score_1 = 2
		elif((len(dif)>=2 and len(dif)<=3 ) and len(dif)!=0):
			score_1 = 1
		else:
			score_1 = 0

		c_errorlist = [c_errorQ1,c_errorQ2,c_errorQ3,c_errorQ4]
		print("c_errorlist:",c_errorlist)
		c_error = max(c_errorlist)

		if(c_error==0):
			score_2 = 2
		elif(c_error<=2 and c_error!=0):
			score_2 = 1
		elif(c_error>=3 and c_error!=0):
			score_2 = 0
		print(point1,point2,point3)
		print("1.Digit(1-12) =",score_1)
		print("2.Digit in wrong quadrant =",score_2)
		print("3.Arrangement and sequencing of the numbers =",score_3)
		font = cv2.FONT_HERSHEY_SIMPLEX
		# org
		org = (5, 50)
		# fontScale
		fontScale = 1
		# Blue color in BGR
		color = (255,255,255)
		# Line thickness of 2 px
		thickness = 2
		text_s1="1.Digit(1-12) ="+str(score_1)
		text_s2="2.Digit in wrong quadrant ="+str(score_2)
		text_s3="3.Arrangement and sequencing of the numbers ="+str(score_3)
		# Using cv2.putText() method
		x,y,z = np.shape(output)
		image_score = np.zeros((x,y,z ), np.uint8)
		image_score = cv2.putText(image_score, text_s1, (5, 50), font, fontScale, color, thickness, cv2.LINE_AA)
		image_score = cv2.putText(image_score, text_s2, (5, 100), font, fontScale, color, thickness, cv2.LINE_AA)
		image_score = cv2.putText(image_score, text_s3, (5, 150), font, fontScale, color, thickness, cv2.LINE_AA)
		h_img = cv2.hconcat([output, image_score])
		print(PATH_TO_RESULT)
		try:
			cv2.imwrite(PATH_TO_RESULT,h_img)
			cv2.imwrite(PATH_TO_ALLRESULT,h_img)
			print("save success!")
		except OSError as error:
			print(error)

		Image.fromarray(h_img).show()

		#reset
		c_errorQ1=0
		c_errorQ2=0
		c_errorQ3=0
		c_errorQ4=0
	except OSError as err:
		print(err)
	
	break
    # return score_1,score_2,score_3



# score_num('test2')




# # Path to image
# PATH_TO_IMAGE = os.path.join(CDT_REWRITE,IMAGE_NAME,IMAGE_NAME+FILE)
# print(PATH_TO_IMAGE)

# #path to save result
# PATH_TO_RESULT = os.path.join(RESULT_FOLDER,IMAGE_NAME,IMAGE_NAME+RES_FILE)
# print(PATH_TO_RESULT)
# #load json file
# data_corr = []
# data_circle = []
# with open("json_num/CDT_rewrite/script_"+IMAGE_NAME+"_num.json") as f:
# 	data = json.load(f)
# for p in data['coordinate']:
# 	data_corr.append(p)
# for p in data['circle']:
# 	data_circle.append(p)

# x,y,r = data_circle
# image = cv2.imread(PATH_TO_IMAGE)
# output = image.copy()
# num = r*(40/100)
# in_r = r-int(num)
# cv2.circle(output,(x, y),in_r, (255,0,0), 3)
# j=2 #เส้นแรกอยู่บนเลข 5 ละวนตามเข็ม
# for i in range(30,385,30):
# 	x2,y2 = draw(x,y,r,i)
# 	xend = int(x2)
# 	yend = int(y2) 
# 	cv2.line(output,(x,y),(xend,yend),(224, 45, 45),3)
# 	line_list.append((xend,yend))

# print("line_list: ",line_list)
# length = len(line_list)
# print("length of line list: ",length)

# for b in range(length):
# 	if (b==11):
# 		p1,p2,p3,p4 = intersec(x,y,*(line_list[b]),*(line_list[0]),in_r)
# 		pointcut = [line_list[0],(p3,p4),(p1,p2),line_list[b]]
# 		listofp.append(pointcut)

# 	else:
# 		p1,p2,p3,p4 = intersec(x,y,*(line_list[b]),*(line_list[b+1]),in_r)
# 		pointcut = [line_list[b+1],(p3,p4),(p1,p2),line_list[b]]
# 		listofp.append(pointcut)

# #load coordinates from json file
# for i in range(0, len(data_corr)):
# 	line.append(data_corr[i])
# 	ymin  = line[i][0]
# 	ymax = line[i][1]
# 	xmin = line[i][2]
# 	xmax = line[i][3]
# 	name = line[i][5][0].split(":")
# 	print(name)
# 	list.append(str(name[0]))
# 	p1 = quadrant(xmin,ymin,x,y)
# 	p2 = quadrant(xmax,ymin,x,y)
# 	p3 = quadrant(xmin,ymax,x,y)
# 	p4 = quadrant(xmax,ymax,x,y)
# 	total = p1+p2+p3+p4
# 	j, k = (xmin+xmax)/2, (ymin+ymax)/2
# 	list_centroid.append((int(j),int(k),str(name[0])))
# 	# Draw a circle in the center of rectangle
# 	cv2.circle(output, center=(int(j), int(k)), radius=3, color=(255, 0, 0), thickness=5)
# 	check_quardrant(name[0],total,xmin)

# match,dif = checklist(list)
# print("match? : ",match)
# print("diff= " ,dif) #คลาสที่ไม่มี 
# for i in list:
# 	print(i)
# 	digit = checkNumberClass(i)
# 	list_digit.append(digit)


# list_digit.sort()
# print(list_digit)

# total_point = 0
# score_1 = 0
# score_2 = 0
# score_3 = 0
# point1 = 0
# point2 = 0
# point3 = 0
# changeclass(dif)
# #print("list of quardrant: ",list_ofq)
# list_ofq.sort(key = lambda x: x[2])  

# # print("sort: ",list_ofq)
# for i in range(0, len(list_ofq)):
# 	print(list_ofq[i][0])

# j = 5
# L=0
# for i in list_centroid:
# 	c1,c2,name = i
# 	c,_ = changestrtoint(c1,c2,name)
# 	sort_list_centroid.append(c)

# sort_list_centroid.sort(key = lambda x: x[2])  
# print("sort: ",sort_list_centroid)
# #start intercept point5
# Y = [5,6,7,8,9,10,11,12,1,2,3,4]
# sort_listofp = [listofp for _,listofp in sorted(zip(Y,listofp))]
# # add class in sort_listofp 
# for k in range(0, len(sort_listofp)):
# 	# print (k)
# 	result = [[m, n, s,t,k+1] for m, n, s,t in sort_listofp]
# 	boxofnum.append(result[k])

# print("boxofnum:",boxofnum)
# # cv2.circle(output,(881, 150),1,0,20)
# # cv2.circle(output,(735, 297),1,0,10)
# # cv2.circle(output,(595, 217),1,0,15)
# # Image.fromarray(output).show()
# #[(710, 126), (639, 197), (530, 134), (556, 38), 1]

# # idx = 12
# # print("idx:",idx)
# # for i in range(len(idx)):
# # 	sort_list_centroid.insert(int(idx[i])-1, (0,0,0))
# print("sort_list_centroid",sort_list_centroid)
# cen=[]
# item=[]
# 	#print("sort_list_centroid:::",sort_list_centroid[])
# 	#print('The index of items is:', sort_list_centroid[:item])

# if(len(boxofnum)>=len(sort_list_centroid)):
# 	size = len(boxofnum)
# if(len(sort_list_centroid)>len(boxofnum)):
# 	size = len(sort_list_centroid)

# for j in range (len(sort_list_centroid)):
	
# 	print("j:",j)
# 	a,b = sort_list_centroid[j][:2]
# 	print("num:",sort_list_centroid[j][2])
# 	print("centriod:",sort_list_centroid[j][:2])
# 	boolean,classnum = checkpoint(boxofnum,sort_list_centroid[j][:2],sort_list_centroid[j][2])
# 		# print(boolean)
# 	try:	
# 		list_boolean.append(boolean)
# 		cv2.putText(output,str(boolean), (int(a),int(b)), font, 1, (116, 7, 97), 2, cv2.LINE_AA)
# 	except OSError as error:
# 		 print(error)  
# 		 Image.fromarray(output).show()
# # while(True):
# #Image.fromarray(output).show()

	
# # for i in range(len(boxofnum)):
# # 	for j in range(len(sort_list_centroid)):
# # 		print(len(sort_list_centroid))
# # 		print("i=",i," j=",j) 
# # 		a,b = sort_list_centroid[j][:2]
# # 		print(a,b,sort_list_centroid[j][2])
# # 		boolean,classnum = checkpoint(boxofnum[i][:4],sort_list_centroid[j][:2],sort_list_centroid[j][2])
# # 		if(boolean==True):
# # 			sort_list_centroid.remove(sort_list_centroid[j])	
# # 		print(sort_list_centroid)
# # for i in range(len(boxofnum)):
# # 	for j in range(len(sort_list_centroid)):
# # 		a, b = sort_list_centroid[j][:2]		
# # 		print("a,b:",a,b)
# # 		print(j,"--------------")
# # 		print(sort_list_centroid[j])
# # 		print(boxofnum[i][4])
# # 		boolean,classnum = checkpoint(boxofnum[i][:4],sort_list_centroid[j][:2],sort_list_centroid[j][2])
# # 		if(boolean==True):
# # 			sort_list_centroid.remove(sort_list_centroid[j])	
# # 			num = (int(a),int(b)) , sort_list_centroid[j][2]
# # 			cen.append(num)
# # 			list_boolean.append(boolean)
# # 		print(sort_list_centroid)
# # 		cv2.circle(output,(int(a),int(b)), radius=0, color=(88, 89, 40), thickness=1)
# 		#cen.append((int(a),int(b)))
# 		#cv2.putText(output,str(boolean), (int(a),int(b)), font, 1, (116, 7, 97), 2, cv2.LINE_AA)
# 		#list_boolean.append(boolean)
# #check out of area of num
# # print(list_boolean)
# # print(cen)
# # for i in range(len(list_boolean)):
# # 	cv2.putText(output,boolean[i], cen[i], font, 1, (116, 7, 97), 2, cv2.LINE_AA)
# # t=[]
# # print(list_boolean)
# # print("list_digit:",list_digit)
# # for i in list_digit:
# # 	print(i)
# # 	print(list_boolean[i])
# # 	t.append(list_boolean[i-1])
# # 	print(t)
# if False in list_boolean :
# 	point3 = 0
# else:
# 	point3 = 1

# #1.clockwise 2.arrange  3.located 1:yes 0:no
# #clockwise
# output,mes = check_clockwise(output,data_corr,x,y,r)
# if(mes=="clockwise"):
# 	point1 = 1
# 	point2 = 1
# elif(mes=="anti-clockwise"):
# 	point1 = 0
# 	point2 = 1
# elif(mes=="other form arrange"):
# 	point1 = 0
# 	point2 = 0


# total_point = point1+point2+point3
# if (total_point==3):
# 	score_3 = 2
# elif (total_point==0):
# 	score_3 = 0
# else:
# 	score_3 = 1

# #test
# #dif = ['1','2','3','6']
# #score rule-1(1-12 on clock) 
# if(len(dif)<=1):
# 	score_1 = 2
# elif((len(dif)>=2 and len(dif)<=3 ) and len(dif)!=0):
# 	score_1 = 1
# else:
# 	score_1 = 0

# #score rule-2(wrong quadrant) 
# # print("c_erroeQ1:",c_errorQ1)
# # print("c_erroeQ2:",c_errorQ2)
# # print("c_erroeQ3:",c_errorQ3)
# # print("c_erroeQ4:",c_errorQ4)

# c_errorlist = [c_errorQ1,c_errorQ2,c_errorQ3,c_errorQ4]
# c_error = max(c_errorlist)

# if(c_error==0):
# 	score_2 = 2
# elif(c_error<=2 and c_error!=0):
# 	score_2 = 1
# elif(c_error>=3 and c_error!=0):
# 	score_2 = 0
# print(point1,point2,point3)
# print("1.Digit(1-12) =",score_1)
# print("2.Digit in wrong quadrant =",score_2)
# print("3.Arrangement and sequencing of the numbers =",score_3)
# font = cv2.FONT_HERSHEY_SIMPLEX
# # org
# org = (5, 50)
# # fontScale
# fontScale = 1
# # Blue color in BGR
# color = (255,255,255)
# # Line thickness of 2 px
# thickness = 2
# text_s1="1.Digit(1-12) ="+str(score_1)
# text_s2="2.Digit in wrong quadrant ="+str(score_2)
# text_s3="3.Arrangement and sequencing of the numbers ="+str(score_3)
# # Using cv2.putText() method
# x,y,z = np.shape(output)
# image_score = np.zeros((x,y,z ), np.uint8)
# image_score = cv2.putText(image_score, text_s1, (5, 50), font, fontScale, color, thickness, cv2.LINE_AA)
# image_score = cv2.putText(image_score, text_s2, (5, 100), font, fontScale, color, thickness, cv2.LINE_AA)
# image_score = cv2.putText(image_score, text_s3, (5, 150), font, fontScale, color, thickness, cv2.LINE_AA)
# h_img = cv2.hconcat([output, image_score])

# try:
# 	cv2.imwrite(IMAGE_NAME,h_img)
# except OSError as error:
# 	print(error)

# Image.fromarray(h_img).show()

# #reset
# c_errorQ1=0
# c_errorQ2=0
# c_errorQ3=0
# c_errorQ4=0

#     # return score_1,score_2,score_3



# # score_num('test2')