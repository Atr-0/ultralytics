import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String, Int32MultiArray
from geometry_msgs.msg import Point, Vector3
from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.yolo.data.dataloaders.v5augmentations import letterbox
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import torch
import cv2
import numpy as np
from ultralytics.yolo.utils.torch_utils import make_divisible
from yolov8 import getbqujieguo, zengqiangduibi1
cmd, jieguo, = "", "",
rclpy.init()


class shibieSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # 订阅者的构造函数和回调函数不需要定时器Timer，因为当收到Message时，就已经启动回调函数了
        # 注意此处不是subscriber，而是subscription
        # 数据类型，话题名，回调函数名，队列长度
        self.subscription = self.create_subscription(
            String, 'shibie', self.listener_callback, 1)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)  # 回调函数内容为打印msg.data里的内容
        global cmd
        cmd = msg.data


Aqu_node = Node("aqu_pub")
Aqu_node_pub = Aqu_node.create_publisher(String, "detect", 10)


def aqu_pub(zhilin):
    global Aqu_node_pub
    msg = String()
    msg.data = zhilin
    # print(zhilin)
    Aqu_node_pub.publish(msg)
    time.sleep(0.03)


Dqu_node = Node("dqu_pub")
Dqu_node_pub = Dqu_node.create_publisher(String, "dqu_detect", 10)


def dqu_pub(zhilin):
    global Dqu_node_pub
    msg = String()
    temp = str(zhilin[0]+"/"+zhilin[1])
    msg.data = temp
    print(zhilin)
    Dqu_node_pub.publish(msg)
    time.sleep(0.03)


def img_mix(img1):
    # img1 = cv2.imread('1.jpg')
    img2 = cv2.imread('/home/zzb/ultralytics/images/tag.jpg')

    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)      # 将图片灰度化
    # ret是阈值（175）mask是二值化图像
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 在img1上面，将logo区域和mask取与使值为0
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

    # 取 roi 中与 mask_inv 中不为零的值对应的像素的值，其他值为 0 。
    # 把logo放到图片当中
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)  # 获取logo的像素信息

    dst = cv2.add(img1_bg, img2_fg)  # 相加即可
    img1[0:rows, 0:cols] = dst
    return img1


def run_aqun(save_path, shibie_subscriber, img_size0=640, stride=32, augment=False, visualize=False, cam=0):
    global cmd, jieguo

    # 读取视频对象: 0 表示打开本地摄像头
    cap = cv2.VideoCapture(cam)

    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    ret_val, img0 = cap.read()
    fps, w, h = 30, img0.shape[1], img0.shape[0]
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 按q退出循环
    while True:
        ret_val, img0 = cap.read()
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        original_image = img0
        # img0=zengqiangduibi1(img0)
        # img0 = enhance_brightness(original_image)
        cv2.line(img0, (602, 0), (558, 322), (0, 0, 255), 4, 8)
        cv2.line(img0, (89, 0), (140, 322), (0, 0, 255), 4, 8)
        cv2.line(img0, (640, 247), (0, 242), (0, 0, 255), 10, 8)
        cv2.imshow('web', img0)
        cv2.waitKey(1)
        if not ret_val:
            break
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)

        if cmd in ["a", "b", "c", "d",]:
            if cmd != "b":
                weights = "/home/zzb/ultralytics/weights/" + cmd + "qu.pt"  # 权重

                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                print(torch.__version__)
                print(torch.cuda.is_available())
                img_size = make_divisible(int(img_size0), int(stride))
                # 导入模型
                weights0 = str(weights[0] if isinstance(
                    weights, list) else weights)
                model = attempt_load_weights(weights if isinstance(weights, list) else weights0,
                                             device=device, inplace=True, fuse=False)

                names = model.names
                jieguo = ""
                if cmd == "d":
                    jieguo = []
                up = ""
                down = ""

                # Padded resize
                img = letterbox(original_image, img_size,
                                stride=stride, auto=True)[0]

                # Convert
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
                img = img[None]     # [h w c] -> [1 h w c]

                # inference
                pred = model(img, augment=augment, visualize=visualize)[0]
                pred = ops.non_max_suppression(
                    pred, conf_thres=0.25, iou_thres=0.7, max_det=1000)

                # plot label
                det = pred[0]
                annotator = Annotator(
                    original_image.copy(), line_width=3, example=str(names))
                if len(det):
                    det[:, :4] = ops.scale_boxes(
                        img.shape[2:], det[:, :4], original_image.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        if cmd == "a" and conf >= 0.8:
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = str(c) + jieguo
                            else:
                                # 下
                                jieguo = jieguo + str(c)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(
                                xyxy, label, color=colors(c, True))
                        elif cmd == "c" and conf > 0.6:
                            print(xyxy)
                            if xyxy[3] < 350:
                                # 上
                                jieguo = "0" + jieguo
                            else:
                                # 下
                                jieguo = jieguo + "1"
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(
                                xyxy, label, color=colors(c, True))
                        elif cmd == "d" and conf >= 0.55:
                            print(xyxy)
                            if xyxy[3] < 350:
                                # 上
                                # jieguo = str((c + 1) + 10) + \
                                #     ("2" if xyxy[0] < 200 else (
                                #         "0" if xyxy[2] > 480 else "1")) + jieguo
                                if up != "":
                                    up = up+"*"
                                up = up + \
                                    (str(
                                        c + 1) + ("2" if xyxy[0] < 200 else ("0" if xyxy[2] > 480 else "1")))
                            else:
                                # 下
                                # jieguo = jieguo + str((c + 1) + 20) + \
                                #     ("2" if xyxy[0] < 200 else (
                                #         "0" if xyxy[2] > 480 else "1"))
                                if down != "":
                                    down = down+"*"
                                down = down + \
                                    (str(
                                        c + 1) + ("2" if xyxy[0] < 200 else ("0" if xyxy[2] > 480 else "1")))
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(
                                xyxy, label, color=colors(c, True))
                im0 = annotator.result()
            else:
                bqu_img = img_mix(original_image)
                bqu = getbqujieguo(bqu_img)
                jieguo = bqu.get_jieguo()
                im0 = bqu.get_plot_image()
                cv2.imshow('webcam:1', bqu.get_det_image())
            cv2.imshow('webcam:0', im0)
            cv2.waitKey(1)
            if cmd == "d":
                jieguo = [up, down]
                dqu_pub(jieguo)
                dqu_pub(jieguo)
            else:
                aqu_pub(jieguo)
                aqu_pub(jieguo)
            cv2.imwrite("/home/zzb/images/shibie/" + cmd +
                        "qu/" + str(time.time()) + ".jpg", im0)
            cv2.imwrite("/home/zzb/images/" + cmd + "qu/" +
                        str(time.time()) + ".jpg", img0)
            cmd = "n"
        if cmd == "n":
            pass
            # aqu_pub(jieguo)
    vid_writer.release()
    cap.release()


def mkdir(mkpath):
    import os
    mkpath = mkpath.strip()
    mkpath = mkpath.rstrip("\\")
    isExists = os.path.exists(mkpath)
    if not isExists:
        os.makedirs(mkpath)
        print(mkpath + ' 创建成功')
        return True
    else:
        return False


def main(args=None):
    imgPath = "/home/zzb/images"
    dePath = "/home/zzb/images/shibie"
    mkdir(imgPath)
    mkdir(dePath)
    for i in ["a", "b", "c", "d"]:
        mkdir(imgPath + "/" + i + "qu")
        mkdir(dePath + "/" + i + "qu")

    shibie_subscriber = shibieSubscriber()
    global cmd, jieguo
    cam = "/dev/camera"
    trycam = 15
    isretry = False
    while rclpy.ok():
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        try:
            if isretry:
                cam = trycam
            run_aqun("/home/zzb/", shibie_subscriber, cam=cam)
        except (AttributeError, cv2.error):
            print("无法打开摄像头，正在尝试" + str(trycam - 1), end="\r")
            isretry = True
            trycam -= 1
        if trycam == 0:
            trycam = 15

        if cmd == "f":
            break
    cv2.destroyAllWindows()
    shibie_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
