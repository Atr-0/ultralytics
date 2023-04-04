import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String
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


def run_aqun(save_path, shibie_subscriber, img_size0=640, stride=32, augment=False, visualize=False, cam=0):
    global cmd, jieguo
    weights = "/home/zzb/ultralytics/weights/" + cmd + "qu.pt"  # 权重

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(torch.cuda.is_available())
    img_size = make_divisible(int(img_size0), int(stride))
    # 导入模型
    weights0 = str(weights[0] if isinstance(weights, list) else weights)
    model = attempt_load_weights(weights if isinstance(weights, list) else weights0,
                                 device=device, inplace=True, fuse=False)

    names = model.names

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
        cv2.imshow('web', img0)
        cv2.waitKey(1)
        if not ret_val:
            break
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.05)
        # print(f'video {frame} {save_path}')
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)

        if cmd in ["a", "c", "d",]:
            jieguo = ""
            # Padded resize
            img = letterbox(img0, img_size, stride=stride, auto=True)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
            img = img[None]     # [h w c] -> [1 h w c]

            # inference
            pred = model(img, augment=augment, visualize=visualize)[0]
            pred = ops.non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            # plot label
            det = pred[0]
            annotator = Annotator(img0.copy(), line_width=3, example=str(names))
            if len(det):
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if conf >= 0.75:
                        if cmd == "a" and conf >= 0.81:
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = str(c) + jieguo
                            else:
                                # 下
                                jieguo = jieguo + str(c)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        elif cmd == "c":
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = "0" + jieguo
                            else:
                                # 下
                                jieguo = jieguo + "1"
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        elif cmd == "d" and conf >= 0.80:
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = str((c + 1) + 10) + \
                                    ("2" if xyxy[0] < 120 else ("0" if xyxy[2] > 480 else "1")) + jieguo
                            else:
                                # 下
                                jieguo = jieguo + str((c + 1) + 20) + \
                                    ("2" if xyxy[0] < 120 else ("0" if xyxy[2] > 480 else "1"))
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            cv2.imshow('webcam:0', im0)
            cv2.waitKey(1)
            aqu_pub(jieguo)
            cmd = "n"
        if cmd == "n":
            print("Amode jieshu")
            break
    # 按q退出循环
    vid_writer.release()
    cap.release()


def run_bqun(save_path, shibie_subscriber, img_size0=640, stride=32, augment=False, visualize=False, cam=0):

    weights = "/home/zzb/ultralytics/weights/bqu.pt"  # 权重

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(torch.cuda.is_available())
    img_size = make_divisible(img_size0, int(stride))
    # 导入模型
    weights0 = str(weights[0] if isinstance(weights, list) else weights)
    model = attempt_load_weights(weights if isinstance(weights, list) else weights0,
                                 device=device, inplace=True, fuse=False)

    names = model.names
    # 读取视频对象: 0 表示打开本地摄像头
    cap = cv2.VideoCapture(cam)
    frame = 0       # 开始处理的帧数

    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    ret_val, img0 = cap.read()
    fps, w, h = 30, img0.shape[1], img0.shape[0]
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # 按q退出循
    while True:
        ret_val, img0 = cap.read()
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('web', img0)
        cv2.waitKey(1)
        if not ret_val:
            break
        frame += 1
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.05)
        # print(f'video {frame} {save_path}')
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        global cmd, jieguo

        if cmd == "b":
            jieguo = ""
            img = letterbox(img0, img_size, stride=stride, auto=True)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
            img = img[None]     # [h w c] -> [1 h w c]

            # inference
            pred = model(img, augment=augment, visualize=visualize)[0]
            pred = ops.non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            # plot label
            det = pred[0]
            annotator = Annotator(img0.copy(), line_width=3, example=str(names))
            up = []
            dowm = []
            if len(det):
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if conf > 0.5:
                        if xyxy[3] < 300:
                            # 上
                            up.append(list([xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls]))
                        else:
                            # 下
                            dowm.append(list([xyxy[0], xyxy[1], xyxy[2], xyxy[3], cls]))
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
            print(up)
            print(dowm)
            for i in dowm:
                if i[4] == 1:
                    print(("xiaceng1"))
                    jieguo = jieguo + "14"
                    break
                else:
                    if (len(dowm) >= 1):
                        if len(dowm) == 2:
                            print('xiacengdouyiyang')
                        elif dowm[0][2] > 500:
                            print("xiaceng0")
                            jieguo = jieguo + "04"
                        elif dowm[0][2] < 500:
                            print("xiaceng2")
                            jieguo = jieguo + "24"
                    else:
                        print("xiacengbushibie")
            for i in up:
                if i[4] == 1:
                    print(("shangceng1"))
                    jieguo = "31" + jieguo
                    break
                else:
                    if (len(up) >= 1):
                        if len(up) == 2:
                            print('shangcengdouyiyang')
                        elif up[0][2] > 500:
                            print("shangceng0")
                            jieguo = "30" + jieguo
                        elif up[0][2] < 500:
                            print("shangceng2")
                            jieguo = "32" + jieguo
                    else:
                        print("shangcengbushibie")
            im0 = annotator.result()
            cv2.imshow('webcam:0', im0)
            aqu_pub(jieguo)
            cv2.waitKey(1)
            cmd = "n"
        if cmd == "n":
            print("bmode jieshu")
            break
    # 按q退出循环
    vid_writer.release()
    cap.release()


def main(args=None):
    shibie_subscriber = shibieSubscriber()
    global cmd, jieguo
    cam = "/dev/camera"
    trycam = 15
    success = False
    while not success:
        try:
            cap = cv2.VideoCapture(cam)
            ret_val, img0 = cap.read()
            fps, w, h = 30, img0.shape[1], img0.shape[0]
            cv2.VideoWriter(
                "/home/zzb/", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cv2.imshow('web', img0)
            cv2.waitKey(1)
            success = True
        except IndexError:
            print("无法打开摄像头，正在尝试", trycam)
            if trycam != 0:
                trycam -= 1

    while rclpy.ok():
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        aqu_pub(jieguo)
        if cmd in ["a", "c", "d",]:
            run_aqun("/home/zzb/", shibie_subscriber, cam=cam)
        if cmd == "b":
            run_bqun("/home/zzb/", shibie_subscriber, cam=cam)
        if cmd == "f":
            break
    cv2.destroyAllWindows()
    shibie_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
