# coding:utf-8
import sys
import cv2
import shutil
import numpy as np
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from tensorflow.keras import optimizers
from MobileNetV2 import MobileNetV2
from readPicture import preprocess_image

data_dict = {
    0:'多云',
    1:'雾',
    2:'雨天',
    3:'雪天',
    4:'晴天',
    5:'雷电'
}

mymodel = MobileNetV2()
mymodel.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
mymodel.load_weights("../weights.h5")

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('images/天气.png'))
        self.setWindowTitle('天气智能识别系统')
        self.to_predict_name = "images/test.jpg"
        self.class_names = ['cloudy','haze','rainy','snowy','sunny','thunder']
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 15)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("天气图片样本")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.jpg", img_show)
        img_init = cv2.resize(img_init, (112, 112))
        cv2.imwrite('images/target.jpg', img_init)
        self.img_label.setPixmap(QPixmap("images/show.jpg"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton(" 上传图片 ")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        video_change = QPushButton(" 本地拍摄 ")
        video_change.clicked.connect(self.call_video)
        video_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        label_result = QLabel(' 天气种类 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('黑体', 17,60))
        self.result.setFont(QFont('楷体', 14))
        label_result_f = QLabel(' 天气置信度 ')
        self.result_f = QLabel("等待识别")
        self.label_info = QTextEdit()
        self.label_info.setFont(QFont('楷体', 12))
        label_result_f.setFont(QFont('黑体', 17,60))
        self.result_f.setFont(QFont('楷体', 14))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.label_info, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(video_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用天气智能识别系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/about.jpg'))
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel('作者：沈木子')
        label_super.setFont(QFont('楷体', 12))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    # 上传图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_single" + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.jpg", img_show)
            img_init = cv2.resize(img_init, (112, 112))
            cv2.imwrite('images/target.jpg', img_init)
            self.img_label.setPixmap(QPixmap("images/show.jpg"))

    #调用摄像头
    def call_video(self):
        cap = cv2.VideoCapture(0)  # 调摄像头，参数为0
        sucess, img = cap.read()  # 读取图片
        cv2.imshow("image", img)  # 展示图片
        k = cv2.waitKey(3000)  # 等待键入，1ms
        if k == 27:  # esc键退出
            cv2.destroyAllWindows()  # 关闭所有窗口
        cv2.imwrite("images/video_image.jpg", img)  # 保存图片
        cv2.destroyAllWindows()
        img_init = cv2.imread('images/video_image.jpg')
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.jpg", img_show)
        img_init = cv2.resize(img_init, (112, 112))
        cv2.imwrite('images/target.jpg', img_init)
        self.img_label.setPixmap(QPixmap("images/show.jpg"))

    # 预测图片
    def predict_img(self):
        i0 = preprocess_image(r'images/target.jpg', 112)
        i1 = tf.expand_dims(i0, 0)
        result1 = np.squeeze(mymodel.predict(i1))
        result2 = tf.convert_to_tensor(result1,dtype=tf.float32)
        result2 = tf.nn.softmax(result2)
        print(result2)
        result2 = { '多云':str(round(result2[0].numpy()*100,2))+'%' ,
                    '雾':str(round(result2[1].numpy()*100,2))+'%' ,
                    '雨天':str(round(result2[2].numpy()*100,2))+'%' ,
                    '雪天':str(round(result2[3].numpy()*100,2))+'%' ,
                    '晴天':str(round(result2[4].numpy()*100,2))+'%' ,
                    '雷电':str(round(result2[5].numpy()*100,2))+'%'
                   }
        result2 = sorted(result2.items(),key=lambda x:x[1],reverse=True)[0:2]
        print(result2)
        self.result_f.setText(result2[0][0]+':'+result2[0][1]+'  '+result2[1][0]+':'+result2[1][1])
        result_index = np.argmax(result1)
        result = data_dict[result_index]
        self.result.setText(result)
        if result == "多云":
            self.label_info.setText("天气多云,不是下雨天,也不是艳阳天,出门淋不着,晒不着！")
        if result == "雾":
            self.label_info.setText(
                "雾天能见度变低,出门一定要注意交通安全,时时刻刻集中注意力，不要走神哦！")
        if result == "雨天":
            self.label_info.setText(
                "下雨路滑，注意安全，记得带伞哦！")
        if result == "雪天":
            self.label_info.setText(
                "天气寒冷，记得添衣防寒！")
        if result == "晴天":
            self.label_info.setText(
                "天气很好，艳阳普照的日子真让人心情舒畅！")
        if result == "雷电":
            self.label_info.setText(
                "雷电天气，注意安全出行！")

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())