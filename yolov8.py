from ultralytics import YOLO
import cv2
import time
import numpy as np
import sys
import similarity
import os
from keras.utils import image_utils as image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2
from image_similarity import *
from skimage.metrics import structural_similarity as compare_ssim
# 计算旋转变换矩阵


def handle_rotate_val(x, y, rotate):
    cos_val = np.cos(np.deg2rad(rotate))
    sin_val = np.sin(np.deg2rad(rotate))
    return np.float32([
        [cos_val, sin_val, x * (1 - cos_val) - y * sin_val],
        [-sin_val, cos_val, x * sin_val + y * (1 - cos_val)]
    ])


def zhengqiang2(img):
    img_t = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_t)
    v2 = np.clip(cv2.add(2 * v, 20), 0, 255)
    img2 = np.uint8(cv2.merge((h, s, v2)))
    img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    return img2

# 图像旋转（以任意点为中心旋转）


def image_rotate(src, rotate=0):
    h, w, c = src.shape
    M = handle_rotate_val(w // 2, h // 2, rotate)
    img = cv2.warpAffine(src, M, (w, h))
    return img


def yuxianpinggu(x, y):
    ans = 0
    for i in range(0, 3):
        ans = ans + (x[i] - y[i])**2
    return ans


def zengqiangduibi(img0):
    a = 2
    b = 0.8
    dst = img0

    for i in range(img0.shape[0]):
        for j in range(img0.shape[1]):
            for c in range(3):
                color = img0[i, j][c] * a + b
                if color > 255:
                    dst[i, j][c] = 255
                elif color < 0:
                    dst[i, j][c] = 0
    return dst


def zengqiangduibi1(img0):
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(20, 20))

    # convert from BGR to LAB color space
    lab = cv2.cvtColor(img0, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img0 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img0


class getbqujieguo():
    def __init__(self, images):
        # Load a model
        # load an official model
        model = YOLO("/home/zzb/ultralytics/weights/cqu.pt")
        # Predict with the model

        # img0=cv2.imread(file_pathname+filename)
        img0 = images
        # img0=zhengqiang2(img0)
        img0 = zengqiangduibi1(img0)
        # img0=log_transfor(img0,-100)
        time.sleep(0.1)
        res = model(img0, conf=0.6)  # predict on an image
        res_plotted = res[0].plot()
        res = list(res)[0]  # get result from generator
        up = []
        down = []
        for i in res.boxes.numpy():
            # print(i.xyxy[0].tolist())
            if i.xyxy[0][1] > 300:
                down.append(i.xyxy[0])
            else:
                up.append(i.xyxy[0])

        # self.plot_image= res_plotted
        upjieguo = []
        simjieguo = []
        self.fabujieguo = ""
        up.sort(key=lambda x: x[0])
        if len(up) == 3:
            juhe = []
            for i in range(3):
                tmp0 = img0[int(up[i][1]):int(up[i][3]),
                            int(up[i][0]):int(up[i][2])]
                # cv2.imwrite("/home/zzb/test1/" + filename[0:len(filename) - 4] + str(i) + str(1) + ".jpg", tmp0)
                tmp1 = cv2.mean(tmp0)
                juhe.append(tmp1)
            for i in range(2, -1, -1):
                tmp0 = img0[int(up[i][1]):int(up[i][3]),
                            int(up[i][0]):int(up[i][2])]
                tmp1 = img0[int(up[i - 1][1]):int(up[i - 1][3]),
                            int(up[i - 1][0]):int(up[i - 1][2])]

                tmp0 = cv2.resize(tmp0, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
                tmp1 = cv2.resize(tmp1, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
                jieguo = similarity.runAllImageSimilaryFun(tmp0, tmp1)
                xiangsidu = yuxianpinggu(juhe[i], juhe[i - 1])

                tmp2 = np.hstack((tmp0, tmp1))
                if i == 0:
                    upjieguo.append(min(jieguo))
                else:
                    upjieguo.append(min(jieguo))

                simjieguo.append(xiangsidu)
            print(juhe)
            print(simjieguo)
            print(upjieguo)
            # jieguozong=[]
            # jieguozong.append(simjieguo,upjieguo,[0,2,1])
            zidian = {0: 0, 1: 2, 2: 1}
            if max(simjieguo) - min(simjieguo) > 600 or max(upjieguo) - min(upjieguo) > 0.05:
                print("666")
                for i in range(2, -1, -1):
                    tmp0 = img0[int(up[i][1]):int(up[i][3]),
                                int(up[i][0]):int(up[i][2])]
                    tmp1 = img0[int(up[i - 1][1]):int(up[i - 1][3]),
                                int(up[i - 1][0]):int(up[i - 1][2])]

                    tmp0 = cv2.resize(tmp0, (224, 224),
                                      interpolation=cv2.INTER_AREA)
                    tmp1 = cv2.resize(tmp1, (224, 224),
                                      interpolation=cv2.INTER_AREA)
                    tmp2 = np.hstack((tmp0, tmp1))

                    if min(simjieguo) == simjieguo[2 - i]:
                        k = zidian[2 - i]
                        print("shangcengcuod shi" + str(k) + "ge")
                        self.fabujieguo = "3" + str(k) + self.fabujieguo
                        # cv2.rectangle(img0,(500,100),(1000,500),(0,255,0),3)
                        cv2.rectangle(img0, (int(up[k][0]), int(up[k][1])),
                                      (int(up[k][2]), int(up[k][3])), (0, 255, 0), 2)
                        # cv2.imwrite("/home/zzb/xiangsi/"+filename[0:len(filename)-4]+str(i)+str(1)+".jpg",tmp2)
                    else:
                        pass
                        # cv2.imwrite("/home/zzb/dif/"+filename[0:len(filename)-4]+str(i)+str(1)+".jpg",tmp2)
            else:
                print("yiyang")
                # cv2.imwrite("/home/zzb/xiangsi/" + filename[0:len(filename) - 4] + str(i) + str(1) + ".jpg", img0)
        downjieguo = []
        simjieguo = []
        down.sort(key=lambda x: x[0])
        if len(down) == 3:
            juhe = []
            for i in range(3):
                tmp0 = img0[int(down[i][1]):int(down[i][3]),
                            int(down[i][0]):int(down[i][2])]
                tmp1 = cv2.mean(tmp0)
                juhe.append(tmp1)
            for i in range(2, -1, -1):
                tmp0 = img0[int(down[i][1]):int(down[i][3]),
                            int(down[i][0]):int(down[i][2])]
                tmp1 = img0[int(down[i - 1][1]):int(down[i - 1][3]),
                            int(down[i - 1][0]):int(down[i - 1][2])]

                tmp0 = cv2.resize(tmp0, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
                tmp1 = cv2.resize(tmp1, (224, 224),
                                  interpolation=cv2.INTER_LINEAR)
                jieguo = similarity.runAllImageSimilaryFun(tmp0, tmp1)
                xiangsidu = yuxianpinggu(juhe[i], juhe[i - 1])

                tmp2 = np.hstack((tmp0, tmp1))
                if i == 0:
                    downjieguo.append(min(jieguo))
                else:
                    downjieguo.append(min(jieguo))
                simjieguo.append(xiangsidu)
            print(juhe)
            print(simjieguo)
            print(upjieguo)
            # jieguozong=[]
            # jieguozong.append(simjieguo,upjieguo,[0,2,1])
            zidian = {0: 0, 1: 2, 2: 1}
            if max(simjieguo) - min(simjieguo) > 0.03 or max(downjieguo) - min(downjieguo) > 0.05:
                print("666")
                for i in range(2, -1, -1):
                    tmp0 = img0[int(down[i][1]):int(down[i][3]),
                                int(down[i][0]):int(down[i][2])]
                    tmp1 = img0[int(down[i - 1][1]):int(down[i - 1][3]),
                                int(down[i - 1][0]):int(down[i - 1][2])]

                    tmp0 = cv2.resize(tmp0, (224, 224),
                                      interpolation=cv2.INTER_AREA)
                    tmp1 = cv2.resize(tmp1, (224, 224),
                                      interpolation=cv2.INTER_AREA)
                    tmp2 = np.hstack((tmp0, tmp1))
                    if min(simjieguo) == simjieguo[2 - i]:
                        k = zidian[2 - i]
                        print("xiacengcuod shi" + str(k) + "ge")
                        self.fabujieguo = self.fabujieguo + str(k) + "4"
                        # cv2.rectangle(img0,(500,100),(1000,500),(0,255,0),3)
                        cv2.rectangle(img0, (int(down[k][0]), int(down[k][1])),
                                      (int(down[k][2]), int(down[k][3])), (0, 255, 0), 2)
                        # cv2.imwrite("/home/zzb/xiangsi/"+filename[0:len(filename)-4]+str(i)+str(1)+".jpg",tmp2)
                    else:
                        pass
                        # cv2.imwrite("/home/zzb/dif/"+filename[0:len(filename)-4]+str(i)+str(1)+".jpg",tmp2)
            else:
                print("yiyang")
                # cv2.imwrite("/home/zzb/xiangsi/"+filename[0:len(filename)-4]+str(i)+str(1)+".jpg",img0)
        self.plot_image = img0

    def get_jieguo(self):
        return self.fabujieguo

    def get_plot_image(self):
        return self.plot_image
