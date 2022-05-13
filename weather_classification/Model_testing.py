import os
import math
import json
from MobileNetV2_with_CBAM import MobileNetV2
# from MobileNetV2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1","Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1 = round((2.0 * Precision * Recall) / (Precision + Recall),3)
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    im_height = 224
    im_width = 224
    batch_size = 16
    def pre_function(img):
        img = img / 255.
        return img
    validation_dir = r'C:\Users\Lenovo\weather\test'
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    model = MobileNetV2()
    # feature.build((None, 224, 224, 3))  # when using subclass model
    pre_weights_path = ''
    model.load_weights(pre_weights_path)

    # read class_indict
    label_path = './class_indices.json'
    assert os.path.exists(label_path), "cannot find {}".format(label_path)
    json_file = open(label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=6, labels=labels)

    # validate
    for step in tqdm(range(math.ceil(total_val / batch_size))):
        val_images, val_labels = next(val_data_gen)
        results = model.predict_on_batch(val_images)
        results = tf.keras.layers.Softmax()(results).numpy()
        results = np.argmax(results, axis=-1)
        labels = np.argmax(val_labels, axis=-1)
        confusion.update(results, labels)
    confusion.plot()
    confusion.summary()