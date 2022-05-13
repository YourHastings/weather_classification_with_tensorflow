import tensorflow as tf
from tensorflow.nn import *
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential,Model

# 通道注意力模块
class CA(Layer):
    def __init__(self,pile_length,ratio=16):
        super(CA,self).__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.dense1 = Dense(int(pile_length/ratio),activation=relu)
        self.dense2 = Dense(int(pile_length),activation=relu)

    def call(self,inputs):
        # 张量运算调用过程(__call__方法调用call)
        input1,input2=inputs,inputs
        # 张量复制两份,分别进行最大池化和平均池化
        max_pool = self.max_pool(input1) # 平均池化
        avg_pool = self.avg_pool(input2) # 最大池化
        # 分别经过两层全连接层
        max_pool_out = self.dense2(self.dense1(max_pool))
        avg_pool_out = self.dense2(self.dense1(avg_pool))
        # Sigmoid激活函数
        out = sigmoid(avg_pool_out + max_pool_out)
        out = Reshape((1,1,out.shape[1]))(out)
        return out

# 空间注意力机制
class SA(Layer):
    def __init__(self):
        super(SA,self).__init__()
        # 定义7×7卷积
        self.conv = Conv2D(1, (7, 7), strides=1, padding='same')

    def call(self, inputs):
        # 张量运算过程
        inputa, inputb = inputs, inputs
        # 张量复制两份,分别沿通道维度进行最大和平均池化
        avg_pool_out = tf.reduce_mean(inputa,axis=3)
        max_pool_out = tf.reduce_max(inputb, axis=3)
        # 合并张量的通道维度
        out = tf.stack([avg_pool_out,max_pool_out],axis=3)
        # 7×7卷积
        out = self.conv(out)
        # Sigmoid激活函数
        out = sigmoid(out)
        return out

# 卷积及其附庸操作
class ConvBNAct(Layer):
    def __init__(self, out_c, k=3, s=1, act = ReLU(max_value=6.0)):
        super(ConvBNAct, self).__init__()
        self.conv = Conv2D(out_c, k,strides=s, padding='SAME',use_bias=False)
        self.bn = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.act = act

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x

# 倒残差模块
class BasicBlock(Layer):
    def __init__(self, in_c, out_c, stride,
                 expand_ratio, kernel, act = ReLU(max_value=6.0)):
        super(BasicBlock, self).__init__()
        self.is_shortcut = stride == 1 and in_c == out_c
        self.med_c = in_c * expand_ratio
        self.ca = CA(out_c)
        self.sa = SA()
        layer_list =[
            # 1x1 PW卷积
            ConvBNAct(self.med_c,1,act=act),
            # 3x3 DW卷积
            DepthwiseConv2D(kernel, stride, padding='SAME',use_bias=False),
            BatchNormalization(momentum=0.9, epsilon=1e-5),act,
            # 1x1 PW卷积
            Conv2D(out_c, 1, strides=1, padding='SAME',use_bias=False),
            BatchNormalization(momentum=0.9, epsilon=1e-5)
        ]
        self.basic_branch = Sequential(layer_list)

    def call(self, inputs, training=False):
        if self.is_shortcut:
            out = self.basic_branch(inputs, training=training)
            # 模块末端添加注意力模块
            out = self.ca(out) * out
            out = self.sa(out) * out
            out = inputs + out
            return out
        else:
            out =  self.basic_branch(inputs, training=training)
            out = self.ca(out) * out
            out = self.sa(out) * out
            return out


#MobileNetV2整体结构
def MobileNet_structure(height,width,kernel,act,classes,config):
    # 输入图像
    input_image = Input(shape=(height, width, 3), dtype='float32')
    # 第一次卷积
    x = ConvBNAct(32, k = kernel, s = 2, act = act)(input_image)
    # 倒残差模块组合
    for _, (t, c, n, s) in enumerate(config):
        for i in range(n):
            stride = s if i == 0 else 1
            x = BasicBlock(x.shape[-1], c, stride,
                           expand_ratio=t,kernel=kernel,act=act)(x)
    # 最后一次卷积
    x = ConvBNAct(1280,k=1,act=act)(x)
    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    # 全连接层
    output = Dense(classes)(Dropout(0.2)(x))
    # 返回MobileNetV2的Model类
    return Model(inputs=input_image, outputs=output)

# MobileNetV2调用函数
def MobileNetV2():
    return  MobileNet_structure(
        height=224, width=224, act = ReLU(max_value=6.0), classes=6, kernel=7,
        config = [
            # [拓展因子,输出通道,重复次数,步长]
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    )