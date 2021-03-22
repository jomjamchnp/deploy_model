import os
import numpy as np
from matplotlib import pyplot as plt 
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
import os.path
import posixpath
ch_eleven = 0
ch_two = 0
ID_NAME = "_h9ezb07yl"

def detect_line(roi,data):
    #Image.fromarray(roi).show()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(roi) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
    #print(lines)
    points = []
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.circle(line_image,(x1,y1),2,(249, 201, 251),10)   
    cv2.circle(line_image,(x2,y2),2,(249, 201, 251),10) 
    # show images
    Image.fromarray(line_image).show()
    return x1,y1,x2,y2

def checkcrash(data_num,x1,y1,x2,y2,typehand,h,k,r):
    list_crash = []
    list_status= []
    list_distance = []
    list_num = [] 
    global ch_eleven,ch_two
    newX1 = x1
    newY1 = y1
    newX2 = x2
    newY2 = y2
    distance = 0
    m = slope(x1,y1,x2,y2)
    i=1
    while(math.pow((newX1-h),2) + math.pow((newY1-k),2) <= math.pow((r),2)):
        newX1+=i         
        newY1 = y1-(m*(x1 - newX1))
        cv2.line(rec,(x1,y1),(int(newX1),int(newY1)),(255,123,45),2)
        status,number=inArea((int(newX1),int(newY1)),data_num)
        distance = math.sqrt(((int(newX1)-x1)**2)+((int(newY1)-y1)**2) )
        if(status):
            crash = (x1,y1,int(newX1),int(newY1),typehand)
            break
    list_distance.append(distance)
    list_crash.append((x1,y1,int(newX1),int(newY1),typehand))
    list_num.append(number)
    list_status.append(status)
    #print(status,number)
    newX1 = x1
    newY1 = y1
    newX2 = x2
    newY2 = y2

    i=-1
    while(math.pow((newX1-h),2) + math.pow((newY1-k),2) <= math.pow((r),2)):
        newX1+=i         
        newY1 = y1-(m*(x1 - newX1))
        cv2.line(rec,(x1,y1),(int(newX1),int(newY1)),(255,123,45),2)
        #print((int(newX),int(newY)))
        status,number=inArea((int(newX1),int(newY1)),data_num)
        distance = math.sqrt(((int(newX1)-x1)**2)+((int(newY1)-y1)**2) )
        if(status):
            crash = (x1,y1,int(newX1),int(newY1),typehand)
            break
    list_distance.append(distance)
    list_crash.append((x1,y1,int(newX1),int(newY1),typehand))
    list_num.append(number)
    list_status.append(status)
    newX1 = x1
    newY1 = y1
    newX2 = x2
    newY2 = y2

    i=1
    while(math.pow((newX2-h),2) + math.pow((newY2-k),2) <= math.pow((r),2)):
        newX2+=i         
        newY2 = y2-(m*(x2 - newX2))
        cv2.line(rec,(x2,y2),(int(newX2),int(newY2)),(5, 107, 120),2)
        #print((int(newX),int(newY)))
        status,number=inArea((int(newX2),int(newY2)),data_num)
        distance = math.sqrt(((int(newX2)-x2)**2)+((int(newY2)-y2)**2) )
        if(status):
            crash = (x2,y2,int(newX2),int(newY2),typehand)
            break
    list_distance.append(distance)
    list_crash.append((x2,y2,int(newX2),int(newY2),typehand))
    list_num.append(number)
    list_status.append(status)
    newX1 = x1
    newY1 = y1
    newX2 = x2
    newY2 = y2
    
    i=-1
    while(math.pow((newX2-h),2) + math.pow((newY2-k),2) <= math.pow((r),2)):
        newX2+=i         
        newY2 = y2-(m*(x2 - newX2))
        status,number=inArea((int(newX2),int(newY2)),data_num)
        cv2.line(rec,(x2,y2),(int(newX2),int(newY2)),(255,123,45),2)
        distance = math.sqrt(((int(newX2)-x2)**2)+((int(newY2)-y2)**2) )
        if(status):
            crash = (x2,y2,int(newX2),int(newY2),typehand)
            break
    list_distance.append(distance)
    list_crash.append((x2,y2,int(newX2),int(newY2),typehand))
    list_num.append(number)
    list_status.append(status)
    x=0
    y=0
    newX=0
    newY=0

    idx = list_distance.index(min(list_distance))
    x,y,newX,newY = list_crash[idx][0],list_crash[idx][1], list_crash[idx][2], list_crash[idx][3]
    print("l",list_status[idx],list_num[idx],list_crash[idx][4])
    cv2.line(rec,(x,y),(newX,newY),(255,123,45),3)  
    if(list_crash[idx][4]=="hour" and list_num[idx]==11):
        ch_eleven = 1
    if(list_crash[idx][4]=="minute" and list_num[idx]==2):
        ch_two = 1
    print("jam ",ch_eleven,ch_two)
    
    if(list_status[idx]==False and list_num[idx]==0 and list_crash[idx][4]=="hour"):
        ch_eleven = check_arrowdegree(x,y,data_num,h,k,r,m,newX,newY)
        #print(x,y,newX,newY)
        #cv2.line(output,(x,y),(newX,newY),(234, 44, 44 ),5)
        #Image.fromarray(output).show()
    print("last ",ch_eleven,ch_two)
    return ch_eleven,ch_two
    
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

 #  check line in number's area n=0 : arrownohead , n=1 : arrow
def inArea(p,data) : #,n,x1,y1,x2,y2
    status = False
    number = 0
    for i in data:
        top_left = (int(i[2]), int(i[0]))
        bottom_right = (int(i[3]), int(i[1]))
        cv2.rectangle(rec, top_left, bottom_right,(53, 77, 206 ), 2) 
        text = str(i[5][0]).split(':')
        check = checkInArea(top_left,bottom_right,p)
        if(check == True):
            status = True
            num_text = str(i[5][0]).split(':')
            #print("true",num_text[0])
            number = checkNumberClass(num_text[0])
    return status,number
    
def checkInArea(top_left,bottom_right,p):
    if (p[0] >= top_left[0] and p[0] <= bottom_right[0] and p[1] >= top_left[1] and p[1] <= bottom_right[1]) :
        return True
    else :
        return False   

def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m

def checkline(h,k,xhead,yhead,xtail,ytail,rec,data_num,r,typehand):
    global ch_eleven,ch_two
    newX = xhead
    newY = yhead
    print(xhead,yhead,xtail,ytail)
    i= -1
    if(xhead>xtail):
        i=1
    m = slope(xhead,yhead,xtail,ytail)
    while(math.pow((newX-h),2) + math.pow((newY-k),2) <= math.pow((r),2)):
        c = xhead-((m)*yhead)
        newX+=i         
        newY = ytail-(m*(xtail - newX))
        cv2.line(rec,(xtail,ytail),(int(newX),int(newY)),(17, 17, 127 ),2)
        #print((int(newX),int(newY)))
        status,number=inArea((int(newX),int(newY)),data_num)
        if(status):
            if(number==11 and typehand=="hour"):
                ch_eleven = 1
            if(number==2 and typehand=="minute"):
                ch_two = 1
            break
    
    # print(status,number,typehand)
    return ch_eleven,ch_two


def checkarrowinbox(h,k,xhead,yhead,xtail,ytail,rec,data_num,r,typehand):
    global ch_eleven,ch_two
    newX = xhead
    newY = yhead
    cv2.circle(rec,(xhead,yhead),2,234,5)
    cv2.circle(rec,(xtail,ytail),2,123,2)
    i= -1
    if(xhead<xtail):
        i=1
    m = slope(xhead,yhead,xtail,ytail)
    while(math.pow((newX-h),2) + math.pow((newY-k),2) <= math.pow((r),2)):
        #print(i)
        c = xhead-((m)*yhead)
        newX+=i         
        newY = ytail-(m*(xtail - newX))
        cv2.line(rec,(xtail,ytail),(int(newX),int(newY)),(223, 222, 39 ),2)
        status,number=inArea((int(newX),int(newY)),data_num)
        if(status):
            if(number==11 and typehand=="hour"):
                ch_eleven = 1
            if(number==2 and typehand=="minute"):
                ch_two = 1
            break
    print(status,number,typehand)
    return ch_eleven,ch_two


def get_values(iterables, key_to_find):
  return list(filter(lambda x:key_to_find in x, iterables)) 

def detect_arrow(img):
    Image.fromarray(img).show()
    k = 1
    list_center = []
    list_dist = []
    list_namemean = []
    # convert image to gray scale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # detect corners with the goodFeaturesToTrack function. 
    corners = cv2.goodFeaturesToTrack(gray, 20, 0.001,7) 
    corners = np.int0(corners) 
    # making a circle at each point that we think is a corner. 
    for i in corners: 
        x, y = i.ravel() 
        center = x,y
        list_center.append(center)
        cv2.circle(img, (x, y), 3, 255, 2)
        #cv2.putText(img,str(k),(x, y), font, 0.1, (0,122,22), 1, cv2.LINE_AA)
        #k = k + 1
    list_diff = []    
    max_corr = []
    #cv2.rectangle(img, (x, y), (x+20, y+20), (255,0,0), 10)
    xmean, ymean = (np.mean(corners, axis = 0)).ravel()
    center = (int(xmean),int(ymean))
    cv2.circle(img,center,1,(72, 45, 238),6)
    #Image.fromarray(img).show()
    for i in list_center:
        x,y = i
        distance = math.sqrt(((xmean-x)**2)+((ymean-y)**2) )
        dist = int(distance)
        max_corr.append((x,y))
        list_dist.append(dist)
    idx = list_dist.index(max(list_dist))
    #print(max(list_dist))
    distance = max(list_dist)
    (p1,p2) = max_corr[idx]
    return distance,(p1,p2),center

def check_boxarrow(rec,box_hand,data_num,h,k,r):
    print(box_hand) #[('hour', (370, 198, 395, 169, 'arrow')), ('miniute', (305, 192, 329, 227, 'arrow'))]
    for i in range(0,len(box_hand)):
        #print(box_hand[i][1][4])
        if(box_hand[i][1][4]=="arrow"):
            checkcrash(data_num,box_hand[i][1][0],box_hand[i][1][1],box_hand[i][1][2],box_hand[i][1][3],box_hand[i][0],h,k,r)
        if(box_hand[i][1][4]=="arrownohead"):
            checkcrash(data_num,box_hand[i][1][0],box_hand[i][1][1],box_hand[i][1][2],box_hand[i][1][3],box_hand[i][0],h,k,r)

def arrownohead(xmin,ymin,lenx,leny,data_corr,rec):
    roi = output[ymin:ymin+leny,xmin:xmin+lenx]
    #Image.fromarray(roi).show()
    x1,y1,x2,y2 = detect_line(roi,data_corr)
    distance = math.sqrt( ((x1-x2)**2)+((y1-y2)**2))
    # cv2.circle(rec,(x1+xmin,y1+ymin),1,(72, 45, 238),5)
    # cv2.circle(rec,(x2+xmin,y2+ymin),1,(72, 45, 238),5)
    #Image.fromarray(rec).show()
    print(distance,x1,y1,x2,y2)
    return distance,x1,y1,x2,y2

def arrow(xmin,ymin,lenx,leny,data_corr,rec):
    roi = output[ymin:ymin+leny,xmin:xmin+lenx]
    distance,(p1,p2),center = detect_arrow(roi) #p1,p2 : tail
    x,y = center
    return distance,(p1,p2),center

def check_data(data,img):
    list_distance = [] 
    list_hands = []
    box_arrow = []
    box_hand = []
    if(len(data)==2):
        for i in range(0, len(data)):
            #print(len(data))
            ymin  = data[i][0]
            ymax = data[i][1]
            xmin = data[i][2]
            xmax = data[i][3]
            #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(212, 238, 127),2)
            name = data[i][5][0].split(":")
            lenx = xmax-xmin
            leny = ymax-ymin
            area = lenx * leny
            #print(area)
            start_point = (data[i][2],data[i][0])
            end_point = (data[i][3],data[i][1])
            #print(name[0])
            if(name[0]=="arrownohead"):
                distance,x1,y1,x2,y2 = arrownohead(xmin,ymin,lenx,leny,data_corr,rec)
                list_distance.append(distance)
                box_arrow.append((x1+xmin,y1+ymin,x2+xmin,y2+ymin,name[0]))
            if(name[0]=="arrow"):     
                distance,(p1,p2),center = arrow(xmin,ymin,lenx,leny,data_corr,rec)
                x,y = center
                list_distance.append(distance)
                box_arrow.append((x+xmin,y+ymin,p1+xmin,p2+ymin,name[0]))

        if(list_distance[0]!=list_distance[1]):
            minute = max(list_distance)
            hour = min(list_distance)
            idx_minute_hand = list_distance.index(minute)
            idx_hour_hand = list_distance.index(hour)
            minute_hand = ("minute",box_arrow[idx_minute_hand]) 
            hour_hand = ("hour",box_arrow[idx_hour_hand]) 
            #check_boxarrow(box_arrow)
            box_hand.append(hour_hand)
            box_hand.append(minute_hand)
            score_4 = 2
        else: 
            score_4 = 1
    else:
        score_4 = 0
    return box_hand,list_hands,score_4
def draw(x1,y1,r,angle):
    length = r
    theta = angle * 3.14 / 180
    #print(theta)
    x2 = x1 + length * cos(theta)
    y2 = y1 + length * sin(theta) 
    return x2,y2
def angle(s1, s2): 
    return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))

def clockwise(data_num,x_center,y_center,radius,angle):
    x1,y1 = x_center,y_center
    eleven_line=[]
    allpoint = []
    an = -angle
    #print(x_center,y_center,radius)
    number_list = []
    for i in range(0,60):
	# check each line
        x2,y2 = draw(x1,y1,radius,an)
        cv2.line(output,(x1,y1),(int(x2),int(y2)),(0,0,0),(1))
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
                status,num = inArea((int(newX),i),data_num)
                #cv2.circle(output, (int(newX),i),2, (0, 128, 255), 2)
                if (status):
                    if (len(num_in_line) == 0):
                        num_in_line.append(num)
                        allpoint.append(((int(newX),i)))
                        if(num==11):
                            eleven = (int(newX),i)
                            eleven_line.append(eleven)
                    else:
                        if (num != num_in_line[len(num_in_line)-1]):
                            num_in_line.append(num)
                            allpoint.append(((int(newX),i)))
                            if(num==11):
                                eleven = (int(newX),i)
                                eleven_line.append(eleven)
        else:
            for i in range(int(a),int(b)):
                newY = (getSlope*i)+ c1
                status,num = inArea((i,int(newY)),data_num)
                #print(status)
                # cv2.circle(output, (i,int(newY)),2, (0, 128, 255), 2)
                if (status):
                    if (len(num_in_line) == 0):
                        num_in_line.append(num)
                        allpoint.append(((i,int(newY))))
                        if(num==11):
                            eleven = (i,int(newY))
                            eleven_line.append(eleven)
                    else:
                        if (num != num_in_line[len(num_in_line)-1]):
                            num_in_line.append(num)
                            allpoint.append(((i,int(newY))))
                            if(num==11):
                                eleven = (i,int(newY))
                                eleven_line.append(eleven)
        if len(num_in_line) != 0:
            number_list.append(num_in_line)
        an = an-6
    print(number_list)
    print(number_list[0])
    # indexes = [index for index in range(len(number_list)) if number_list[index] == '[11]']
    if(number_list[0]==[11]):
        check = 1
        indices = [i for i, x in enumerate(number_list) if x == [11]]
        print(allpoint[max(indices)+1])
        next_point = allpoint[len(number_list)-1]
        print("eleven_line:",eleven_line)
        # for i in eleven_line:
        #     print(i)
        #     cv2.circle(output,(i[0],i[1]),3,(236, 130, 30),5)
        average = [sum(x)/len(x) for x in zip(*eleven_line)]
        print(average)
        point_eleven = average
        cv2.circle(output,(int(average[0]),int(average[1])),3,(126, 236, 80 ),5)
        cv2.circle(output,(int(next_point[0]),int(next_point[1])),3,(126, 236, 80 ),5)
    else:
        check=0
        point_eleven=0
        next_point=0
    status = 0

    return point_eleven,next_point,check

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

def check_arrowdegree(x1,y1,data_num,h,k,r,m,x2,y2):#x,y,data_num,h,k,r,m,newX,newY
    ch_eleven = 0
    newX,newY = 0,0
    for i in data_num:
         cv2.rectangle(output, (i[2], i[0]),(i[3], i[1]), (0, 255, 255), 2)
    print("--check_arrowdegree--")
    cv2.line(output,(x1,y1),(x2,y2),(184,158,170),3)
    m = slope(x1,y1,x2,y2)
    newX = newX-1         
    newY = y1-(m*(x1 + newX))
    dist = math.sqrt((x1 - newX)**2 + (y1 - newY)**2)
    x3,y3 = draw(x1,y1,dist,0)
    #cv2.line(output,(x1,y1),(int(x3),int(y3)),(29, 120, 15 ),2)
    
    get_angle = getAngle((x2,y2), (x1, y1), (x3,y3))
    print('Angle = ',get_angle)
    x4,y4 = draw(x1,y1,dist,-get_angle)
    point_1,point_2,check = clockwise(data_num,x1,y1,r,get_angle)
    if(check==1):
        ang_11andhand = getAngle(point_1,(x1, y1),(x2,y2))
        ang_11andnext = getAngle((x2,y2),(x1, y1),point_2)

        print("ss",ang_11andhand,ang_11andnext) 
        if(ang_11andhand<ang_11andnext):
            ch_eleven = 1
            print("correct")
        else:
            ch_eleven = 0
    x5,y5 = draw(x1,y1,dist,-ang_11andhand)
    print(point_1)
    cv2.line(output,(x1,y1),(int(point_1[0]),int(point_1[1])),(0, 9, 69),2)

    x6,y6 = draw(x1,y1,dist,-ang_11andnext)
    cv2.line(output,(x1,y1),(int(point_2[0]),int(point_2[1])),(102, 198, 42 ),2)

    cv2.line(output,(x1,y1),(int(x4),int(y4)),(120, 156, 237 ),2)
    
    # m = slope(x1,y1,x2,y2)
    # for i in range(0,60):
    #     #cv2.circle(output, (x2,y2),2, (0, 128, 255), 20)
    #     dist = math.sqrt((x1 - h)**2 + (y1 - k)**2)
    #     #cv2.line(output,(h,k),int(h+dist,k+dist),(255,123,45),2)
    #     x2,y2 = draw(x1,y1,abs(r-dist),an)
    #     #cv2.line(output,(x1,y1),(int(x2),int(y2)),(255,123,45),2)
    #     cv2.circle(output,(int(x2),int(y2)),2, (0, 128, 255), 10)
    #     an=an+6
    # for i in range(0,60):

    #     #m = slope(xhead,yhead,xtail,ytail)
    #     while(math.pow((x2-h),2) + math.pow((y2-k),2) <= math.pow((r),2)):
    #         c = x1-((m)*y1)
    #         newX+=i         
    #         newY = y2-(m*(x2 - newX))
    #         cv2.line(output,(int(x2),int(y2)),(int(newX),int(newY)),(223, 222, 39 ),2)
    #         status,number=inArea((int(newX),int(newY)),data_num)
    #         if(status):
    #             print("number:",number)
    #             break
    #          x2,y2 = draw(x,y,r,i)
    # #print(status,number,typehand)
    #     #x2,y2 = draw(x1,y1,r,an)
            
    #     #cv2.line(output,(x1,y1),(int(x2),int(y2)),(255,123,45),(1))
    #     an=an-5
        
	# check each line
    #     x2,y2 = draw(x,y,r,an)
    #     cv2.line(output,(x1,y1),(int(x2),int(y2)),(255,123,45),(1))
    #     getSlope = slope(x1,y1,x2,y2)
    #     c1 = y1-(getSlope*x1)
    #     a,b = x1,x2
    #     c,d = y1,y2
    #     if (x1 < x2):
    #         a = x1
    #         b = x2
    #     elif (x > x2):
    #         a = x2
    #         b = x1
    #     if (y < y2):
    #         c = y1
    #         d = y2
    #     elif (y > y2):
    #         c = y2
    #         d = y1
        
    #     num_in_line = []
    # # check each point 90 and 270
    #     if an == 90 or an == 270:
    #         for i in range(int(c),int(d)):
    #             newX = (i - c1)/getSlope
    #             status,num = inArea((int(newX),i),data_num)
    #             # cv2.circle(output, (int(newX),i),2, (0, 128, 255), 2)
    #             if (status):
    #                 if (len(num_in_line) == 0):
    #                     num_in_line.append(num)
    #                 else:
    #                     if (num != num_in_line[len(num_in_line)-1]):
    #                         num_in_line.append(num)
    #     else:
    #         for i in range(int(a),int(b)):
    #             newY = (getSlope*i)+ c1
    #             status,num = inArea((i,int(newY)),data_num)
    #             # cv2.circle(output, (i,int(newY)),2, (0, 128, 255), 2)
    #             if (status):
    #                 if (len(num_in_line) == 0):
    #                     num_in_line.append(num)
    #                 else:
    #                     if (num != num_in_line[len(num_in_line)-1]):
    #                         num_in_line.append(num)
    #     if len(num_in_line) != 0:
    #         number_list.append(num_in_line)
    #     an = an+6
    #     break
    #for i in range(0,60):
    # check each line
        # while (math.pow((i-x),2) + math.pow((newY-y),2) <= math.pow((r),2)):
        # x2,y2 = draw(x,y,r,an)
        # cv2.line(output,(x,y),(int(x2),int(y2)),(235, 181, 206),(1))
        # c1 = y1-(m*x1)
        # num_in_line = []
        # newY = (m*i)+ c1
        # status,num = inArea((i,int(newY)),data_num)
        # # cv2.circle(output, (i,int(newY)),2, (0, 128, 255), 2)
        # if (status):
        #     if (len(num_in_line) == 0):
        #         num_in_line.append(num)
        #     else:
        #         if (num != num_in_line[len(num_in_line)-1]):
        #             num_in_line.append(num)
        # an = an-6
    # # check each point 90 and 270
    #     if an == 90 or an == 270:
    #         for i in range(int(c),int(d)):
    #             newX = (i - c1)/getSlope
    #             status,num = inArea((int(newX),i),data)
    #             # cv2.circle(output, (int(newX),i),2, (0, 128, 255), 2)
    #             if (status):
    #                 if (len(num_in_line) == 0):
    #                     num_in_line.append(num)
    #                 else:
    #                     if (num != num_in_line[len(num_in_line)-1]):
    #                         num_in_line.append(num)
    #     else:
        # for i in range(int(a),int(b)):

        # if len(num_in_line) != 0:
        #     number_list.append(num_in_line)
        #an = an+6
    # num_in_line=[]
    # i=-1
    # for i in range(int(a),int(b)):
	# 			newY = (m*i)+ c1
	# 			status,num = inArea((i,int(newY)),data)
	# 			# cv2.circle(output, (i,int(newY)),2, (0, 128, 255), 2)
	# 			if (status):
	# 				if (len(num_in_line) == 11):
	# 					num_in_line.append(num)
	# 				else:
	# 					if (num != num_in_line[len(num_in_line)-1]):
	# 						num_in_line.append(num)
    # for i in range(15,385,15):
    #     x2,y2 = draw(h,k,r,i)
    #     xend = int(x2)
    #     yend = int(y2) 
    #     cv2.line(output,(x,y),(xend,yend),(235, 181, 206),3)
    #     status,number=inArea((int(xend),int(yend)),data_num)
    #     if(status==True):
    #         break
        
    Image.fromarray(output).show()
    return ch_eleven
        #line_list.append((xend,yend))



# def score_hand(name):    
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = '_h9ezb07yl'
FILE = '_hands.jpg'
IMAGE_FOLDER = 'image_test'

# Grab path to current working directory
CWD_PATH = os.getcwd()
PREVIOS_PATH = os.path.abspath(CWD_PATH+ "/../")
# Path to image
PATH_TO_IMAGE = os.path.join(IMAGE_FOLDER,IMAGE_NAME+FILE)
#Read image 
image = cv2.imread(PATH_TO_IMAGE)

#read hand
with open("json_hand/script"+IMAGE_NAME+".json") as f:
    data = json.load(f)

data_corr = []
data_circle = []
# read num from json
with open("json_num/script"+IMAGE_NAME+".json") as f:
    data2 = json.load(f)
for p in data2['coordinate']:
    data_corr.append(p)
for p in data2['circle']:
    data_circle.append(p)

#load coordinates from json file
name=[]
list_point=[] #list of coordinate no head [(x1,y2),(x2,y2)]
list_centroid =[]
list_crashnum = []
hour_hand = 0
minute_hand = 0
output = image.copy()
rec = image.copy()
font=cv2.FONT_ITALIC
#data[0][0] = ymin , data[0][1]=ymax, data[0][2]=xmin, data[0][3]=xmax
h,k,r = data_circle
#score
score_4 = 0
score_5 = 0

box_hand,list_hands,score_4 = check_data(data,rec)
check_boxarrow(rec,box_hand,data_corr,h,k,r)
#rule4    

#rule5
if(ch_eleven==1):
    score_5 = score_5 + 1
if(ch_two==1):
    score_5 = score_5 + 1

print("4.have 2 hands: ",score_4)
print("5.hands on correct digit:",score_5)
Image.fromarray(rec).show()
cv2.imwrite(os.path.join(IMAGE_FOLDER,ID_NAME+'frame.jpg'),rec)
    # return score_4,score_5

# score_hand('_vgh42ixnj')