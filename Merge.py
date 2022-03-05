import torch
from torch.autograd import Variable as V
import os
import numpy as np
import matplotlib.image as mpimg
import math
import cv2
from time import time

from networks.dinknet import DinkNet34

BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)



        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

source = 'submit/'
val = os.listdir(source)
solver = TTAFrame(DinkNet34)

solver.load('weights/first.th')
target = 'submits/'
if os.path.exists(target):
    pass
else:
    os.mkdir(target)
for i,name in enumerate(val):
    print("pic name",name)
    mask = solver.test_one_img_from_path(source+name)
    mask[mask>4.0] = 255
    mask[mask<=4.0] = 0
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    cv2.imwrite('submits/mask.png',mask.astype(np.uint8))



def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]

def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 8

    min_angle_to_merge = 15

    for line in lines:
        create_new_group = True
        group_updated = False
        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))
                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        group.append(line)
                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break
        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        new_group.append(line2)

            super_lines.append(new_group)
    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final

def merge_lines_segments1(lines, use_log=False):
    if(len(lines) == 1):
        return lines[0]
    line_i = lines[0]
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])
    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        points = sorted(points, key=lambda point: point[1])

        if use_log:
            print("use y")
    else:
        points = sorted(points, key=lambda point: point[0])
        if use_log:
            print("use x")
    return [points[0], points[len(points)-1]]

#求斜边长度函数，判断两条直线是否相据很近，可调参
def lines_close(line1, line2):
    dist1 = math.hypot(line1[0][0] - line2[0][0], line1[0][0] - line2[0][1])
    dist2 = math.hypot(line1[0][2] - line2[0][0], line1[0][3] - line2[0][1])
    dist3 = math.hypot(line1[0][0] - line2[0][2], line1[0][0] - line2[0][3])
    dist4 = math.hypot(line1[0][2] - line2[0][2], line1[0][3] - line2[0][3])

    if (min(dist1,dist2,dist3,dist4) < 100):
        return True
    else:
        return False
#求线段长度函数
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def DistancePointLine(px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = lineMagnitude(px,py,x1,y1)
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):

        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:

        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1],
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1],
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])


    return min(dist1,dist2,dist3,dist4)

def DistantPoint(point1,point2):
    l = int(math.sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2)))
    return l


def PointtoPoint(all_lines_second):
    for i in range(len(all_lines_second)):
        oneline_first = all_lines_second[i]
        for j in range(i+1,len(all_lines_second)):
            oneline_second=all_lines_second[j]
            mindis = min(DistantPoint(oneline_first[0], oneline_second[0]),
                         DistantPoint(oneline_first[0], oneline_second[1]),
                         DistantPoint(oneline_first[1], oneline_second[0]),
                         DistantPoint(oneline_first[1], oneline_second[1]))
            for i in range(4):
                if DistantPoint(oneline_first[math.floor(i/2)], oneline_second[i % 2]) == mindis:
                    if mindis < 10:
                        oneline_first[math.floor(i/2)] = oneline_second[i % 2]
                    break
    return all_lines_second

def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def isEnd(thisline,thispoint,all_lines):
    for line in all_lines:
        if thisline!=line:
            for i in range(4):
                if (thisline[math.floor(i/2)]==line[i%2])&(thisline[math.floor(i/2)]==thispoint):
                    return False
    return True

def isPointonLine(px,py,line):
    max_distance=8
    lines=[]
    x1 = line[0][0]  # 取四点坐标
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]
    if (px-x1)*(px-x2)<0 and (py-y1)*(py-y2)<0:
        line1=[(x1,y1),(int(px),int(py))]
        line2=[(int(px),int(py)),(x2,y2)]
        lines.append(line1)
        lines.append(line2)
        return True ,lines
    elif lineMagnitude(px,py,x1,y1)<max_distance :
        line1=[(int(px),int(py)),(x2,y2)]
        lines.append(line1)
        return True, lines
    elif lineMagnitude(px,py,x2,y2)<max_distance:
        line1 = [(x1, y1), (int(px),int(py))]
        lines.append(line1)
        return True, lines
    return  False ,lines

def cross_point(line1, line2):  # 计算交点函数
    # 是否存在交点
    point_is_exist = False
    lines=[]
    x = 0
    y = 0
    x1 = line1[0][0]  # 取四点坐标
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]
    for i in range(4):
        if line1[math.floor(i/2)]==line2[i%2]:
            return point_is_exist,lines
    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
            flag1,lines1 =isPointonLine(x,y,line1)
            flag2, lines2 = isPointonLine(x, y, line2)
            if flag1 and flag2:
                lines.extend(lines1)
                lines.extend(lines2)
                point_is_exist = True
    elif k2 is None:

        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
        flag1, lines1 = isPointonLine(x, y, line1)
        flag2, lines2 = isPointonLine(x, y, line2)
        if flag1 and flag2:
            lines.extend(lines1)
            lines.extend(lines2)
            point_is_exist = True
    return point_is_exist, lines
#返回bool，和点坐标

def CorssMerge(all_lines):
    templines=all_lines
    while(True):
        Flag = False
        for i in range(len(templines)-1):
            firstline=templines[i]
            for j in range(i+1,len(templines)):
                secondline=templines[j]
                a,lines=cross_point(firstline,secondline)
                if a:
                    templines.remove(firstline)
                    templines.remove(secondline)
                    for line in lines:
                        if DistantPoint(line[0], line[1]) > 5:
                            templines.append(line)
                    Flag=True
                    break
            if Flag:
                break
        if i==(len(templines)-2) and j==(len(templines)-1):
            break
    ttemp=[]
    for line in templines:
        if DistantPoint(line[0],line[1])>8:
            ttemp.append(line)
    return ttemp

def process_lines(image_src):

#H函数预处理-----------------------------------------------------------------
    img = cv2.imread(image_src)
    KsizeX = 5  # x方向核函数尺寸，取值为正奇数
    KsizeY = 3  # y方向核函数尺寸，可与X方向不同
    img = cv2.GaussianBlur(img, (KsizeX, KsizeY), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.bitwise_not(thresh1)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 20, minLineLength=5, maxLineGap=11.45)
#预处理结束----------------------------------------------------------------
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])

    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)
    print("process groups lines", len(_lines), len(merged_lines_all))
    img_merged_lines = mpimg.imread(image_src)
    all_lines_second=PointtoPoint(CorssMerge(PointtoPoint(merged_lines_all)))
#写操作
#----------------------------------------------------------------------------
    filename = 'submits/data.txt'
    with open(filename, 'w') as f:
        f.truncate()
    for line in all_lines_second:
        x1=line[0][0]
        y1=line[0][1]
        x2=line[1][0]
        y2=line[1][1]
        cv2.line(img_merged_lines, (x1, y1), (x2, y2), (0,0,255), 1)
        l = int(math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)))
        strings = '(' + str(x1) + ', ' + str(y1) + ')' + ' ' + '(' + str(x2) + ', ' + str(y2) + ')' + ' ' + str(l)
        with open(filename, 'a') as f:
            f.write(strings)
            f.write('\n')
    cv2.imwrite('submits/merged_lines.jpg',img_merged_lines)


    return merged_lines_all
#----------------------------------------------------------------------------

filename='submits/mask.png'
process_lines(filename)