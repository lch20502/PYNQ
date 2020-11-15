import numpy as np
import collections
import cv2
import numba as nb

class ANN_net():
    def __init__(self):
        #通过调用OpenCV函数创建ANN
        self.animals_net = cv2.ml.ANN_MLP_create()
        
        #ANN_MLP_RPROP和ANN_MLP_BACKPROP都是反向传播算法，此处设置相应的拓扑结构
        self.animals_net.setLayerSizes(np.array([100, 50, 2]))
        self.animals_net.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
        self.animals_net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
        
        #指定ANN的终止条件
        self.animals_net.setTermCriteria(( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 ))

    def train():
        records = []
        RECORDS = 5000
        ALL_sample = np.zeros((RECORDS,3), dtype=np.float32)
        ALL_class = np.zeros((RECORDS,2), dtype=np.float32)
        data = Data_maker
        for x in np.arange(0, RECORDS, 2):   #产生数据集
            None

        EPOCHS = 50  #训练次数
        for e in range(0, EPOCHS):
            self.animals_net.train(ALL_sample, cv2.ml.ROW_SAMPLE, ALL_class)

    def forward(x):
        return self.animals_net.predict(x)[0]  #返回 0或1

def getColorList():
    dict = collections.defaultdict(list)
 
    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
 
    # #灰色
    if 0:
        lower_gray = np.array([0, 0, 46])
        upper_gray = np.array([180, 43, 220])
        color_list = []
        color_list.append(lower_gray)
        color_list.append(upper_gray)
        dict['gray']=color_list
 
    # 白色
    if 0:
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        dict['white'] = color_list
 
    #红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list
 
    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list
 
    #橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([19, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list
 
    #黄色
    lower_yellow = np.array([20, 150, 156])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
 
    #绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list
 
    #青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list
 
    #蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list
 
    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list
 
    return dict
 
#处理图片
def get_color(frame):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1],)
        #cv2.imwrite(d+'.jpg',mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)

        binary_mid = binary.copy()
        contours, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        
        sum = 0
        for c in contours:
            sum+=cv2.contourArea(c)

        if d == 'red':   #red和red2算在一起
            mask = cv2.inRange(hsv,color_dict['red2'][0],color_dict['red2'][1],)
            #cv2.imwrite(d+'.jpg',mask)
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary,None,iterations=2)

            binary_mid = binary.copy()
            contours, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                sum+=cv2.contourArea(c)
        elif d == 'red2':
            continue

        if sum > maxsum :
            maxsum = sum
            color = d

    return color  #,frame_copy

def getColorList2():
    dict = collections.defaultdict(list)
 
    # 黑色
    if 0:
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 46])
        color_list = []
        color_list.append(lower_black)
        color_list.append(upper_black)
        dict['black'] = color_list
 
    # #灰色
    if 0:
        lower_gray = np.array([0, 0, 46])
        upper_gray = np.array([180, 43, 220])
        color_list = []
        color_list.append(lower_gray)
        color_list.append(upper_gray)
        dict['gray']=color_list
 
    # 白色
    if 0:
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        dict['white'] = color_list
 

    # 红2、橙、黄、绿、青、蓝、紫色、红
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([77, 255, 255])   #到绿为止
    #upper_red = np.array([180, 255, 255])    #到红为止
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list
 
    return dict
 
#处理图片
def get_color2(frame):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList2()
    for d in color_dict:
        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
        cv2.imwrite(d+'.jpg',mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)

        binary_mid = binary.copy()
        contours, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        sum = 0
        for c in contours:
            sum+=cv2.contourArea(c)
        if sum > maxsum :
            maxsum = sum
            color = d
 
    if 1:  #获取最大值时的边界
        mask = cv2.inRange(hsv,color_dict[color][0],color_dict[color][1])
        cv2.imwrite(color+'.jpg',mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)
        binary_mid = binary.copy()
        contours, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    return color,contours

def frame_deal(contours):
    mid = []
    for i in contours:
        for j in i:
            mid.append(j)
    contours_frame = np.asarray(mid)
    contours_frame = contours_frame[:,0,:]

    c_x = np.sort(contours_frame[:,0],axis = 0)  #由小到大排序
    c_y = np.sort(contours_frame[:,1],axis = 0) 

    mid_xmin , mid_xmax , mid_ymin , mid_ymax =  c_x[10] , c_x[-10] , c_y[10] , c_y[-10]   #去掉最近10个最大最小
    
    contours_frame = np.array([[[ mid_xmin , mid_ymin ],[ mid_xmax , mid_ymin ],[ mid_xmax , mid_ymax ],[ mid_xmin , mid_ymax ]]])
    return contours_frame

def get_color3(frame,color_fruit):
    #print('go in get_color')
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    contours_final = []
    color_dict = getColorList3()
    for d in color_dict:
        if d == color_fruit:   #跳过对水果色的判断
            continue

        mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1],)
        #cv2.imwrite(d+'.jpg',mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary,None,iterations=2)   

        binary_mid = binary.copy()
        contours, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
        
        sum = 0
        for c in contours:
            sum += cv2.contourArea(c)
            

        if d == 'red':   #red和red2算在一起
            mask = cv2.inRange(hsv,color_dict['red2'][0],color_dict['red2'][1],)
            #cv2.imwrite(d+'.jpg',mask)
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary,None,iterations=2)

            binary_mid = binary.copy()
            contours1, hierarchy = cv2.findContours(binary_mid,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours1:
                sum+=cv2.contourArea(c)
            contours = contours + contours1   #shit
        elif d == 'red2':
            continue

        if sum > maxsum :
            maxsum = sum
            color = d
            contours_final = contours

    if color:
        max_area = -100
        mid = 0
        k = 0
        for c in contours_final:
            if cv2.contourArea(c) > max_area:
                max_area = cv2.contourArea(c)
                k = mid 
            mid += 1

    return color,contours_final,k

def getColorList3():
    dict = collections.defaultdict(list)
 
    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list
 
    # #灰色
    if 0:
        lower_gray = np.array([0, 0, 46])
        upper_gray = np.array([180, 43, 220])
        color_list = []
        color_list.append(lower_gray)
        color_list.append(upper_gray)
        dict['gray']=color_list
 
    # 白色
    if 0:
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 30, 255])
        color_list = []
        color_list.append(lower_white)
        color_list.append(upper_white)
        dict['white'] = color_list
 
    if 0:
        #红色
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red'] = color_list
    
        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red2'] = color_list
    
        #橙色
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([11, 255, 255])
        color_list = []
        color_list.append(lower_orange)
        color_list.append(upper_orange)
        dict['orange'] = color_list
    
        #黄色
        lower_yellow = np.array([12, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        color_list = []
        color_list.append(lower_yellow)
        color_list.append(upper_yellow)
        dict['yellow'] = color_list
    
        #绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        color_list = []
        color_list.append(lower_green)
        color_list.append(upper_green)
        dict['green'] = color_list
    
        #青色
        lower_cyan = np.array([78, 43, 46])
        upper_cyan = np.array([99, 255, 255])
        color_list = []
        color_list.append(lower_cyan)
        color_list.append(upper_cyan)
        dict['cyan'] = color_list
    
        #蓝色
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        color_list = []
        color_list.append(lower_blue)
        color_list.append(upper_blue)
        dict['blue'] = color_list
    
        # 紫色
        lower_purple = np.array([125, 43, 46])
        upper_purple = np.array([155, 255, 255])
        color_list = []
        color_list.append(lower_purple)
        color_list.append(upper_purple)
        dict['purple'] = color_list
 
    return dict

def square(clist):
    x=2
    y=4
    z=len(clist)
    slist=[]
    for i in contours_bad:
        sudden=np.zeros((4,1,2),dtype=np.int32)
        x_max=np.max(i[:,:,0])
        x_min=np.min(i[:,:,0])
        y_max=np.max(i[:,:,1])
        y_min=np.min(i[:,:,1])
        sudden[0][0][0]=x_max
        sudden[0][0][1]=y_max
        sudden[1][0][0]=x_max
        sudden[1][0][1]=y_min
        sudden[2][0][0]=x_min
        sudden[2][0][1]=y_min
        sudden[3][0][0]=x_min
        sudden[3][0][1]=y_max
        slist.append(sudden)
    return slist



picture_path = r'C:\PICTURE\apple\banana_test1.jpg'
threshold = 0.1
threshold1 = 40
bad_sign = False


img = cv2.imread(picture_path)

model = ANN_net()

if 0:
    h , w , _ = img.shape
    h -= h%4
    w -= w%4
    img = cv2.resize(img,(720,540))
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#img = img[200:-200,800:-400]
color = get_color(img)  #获取准确的颜色分类

#物体识别
_ , contours = get_color2(img)  #返回背景色以外的颜色边界 和 边界 

#瑕疵识别
color_bad , contours_bad  ,k= get_color3(img,color)

contours_bad = square(contours_bad)

if color_bad :       #如果判断苹果是坏的，则在对应区域画图，否则退出循环
    bad_sign = True 
    cv2.drawContours(img,[contours_bad[k]],-1,(0,255,0),3)
    #cv2.drawContours(img,contours_bad,-1,(0,255,0),3)

print(contours_bad[k][0,0,0]-contours_bad[k][2,0,0],contours_bad[k][0,0,1]-contours_bad[k][2,0,1])

contours = frame_deal(contours)   #将水果边界改为方框
cv2.drawContours(img,contours,-1,(0,0,255),3)  #给最终彩图水果边界加边框

if bad_sign:
    print('水果颜色：',color,'。坏了,坏的地方颜色是：',color_bad,'。')
else:
    print('水果颜色：',color,'。没坏。')

if 1:
    #cv2.imshow('s',hsv1)
    cv2.imshow('wname',img)
    #cv2.imshow('wname',img_grad)
    #cv2.imshow('wname',img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.destroyWindow('wname')
