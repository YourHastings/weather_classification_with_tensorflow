import tensorflow as tf
import csv
from tensorflow.keras import optimizers
from readPicture import preprocess_image,readpic
from MobileNetV2_with_CBAM import MobileNetV2
# from MobileNetV2 import MobileNetV2

loss_list = []
accuracy_list = []
max_accuracy_list = []

def load_picture(path, label):
    return preprocess_image(path, 224), label

# 设置单机多卡分布式训练策略
mystrategy = tf.distribute.MirroredStrategy()
single_batch_size = 32
all_batchsize = single_batch_size * mystrategy.num_replicas_in_sync

# 读取图片位置和类别标签
# 数据下载地址：http://vcc.szu.edu.cn/research/2017/RSCM.html
# 下载后参考split_data.py中方法划分数据集为训练集、验证集、测试集
path0,label0=readpic(r'C:\Users\Lenovo\weather\train')
path0,label0=path0[0:],label0[0:]
path1,label1=readpic(r'C:\Users\Lenovo\weather\val')
path1,label1=path1[0:],label1[0:]

# 数据集构建、预处理、散列、分批
dbtrain = tf.data.Dataset.from_tensor_slices((path0, label0))
dbval = tf.data.Dataset.from_tensor_slices((path1, label1))
dbtrain = dbtrain.map(load_picture).shuffle(len(path0)).batch(all_batchsize)
dbval = dbval.map(load_picture).shuffle(len(path1)).batch(all_batchsize)

# 分布式数据集构建
dbtrain = mystrategy.experimental_distribute_dataset(dbtrain)
dbval = mystrategy.experimental_distribute_dataset(dbval)

# 定义损失函数
with mystrategy.scope():
    model_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    def compute_loss(labels,predictions):
        per_loss = model_loss(labels,predictions)
        return tf.nn.compute_average_loss(per_loss, global_batch_size= all_batchsize)

# 定义衡量指标
with mystrategy.scope():
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

# 创建模型、优化器
with mystrategy.scope():
    model = MobileNetV2()
    optimizer = optimizers.Adam(learning_rate=5e-4)

# 训练步骤
def train_step(inputs):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_acc.update_state(labels, predictions)
    return loss

# 测试步骤
def test_step(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    t_loss = model_loss(labels, predictions)
    val_loss.update_state(t_loss)
    val_acc.update_state(labels, predictions)

@tf.function
def distributed_train_step(data_inputs):
    per_losses = mystrategy.run(train_step,args=(data_inputs,))
    return mystrategy.reduce(tf.distribute.ReduceOp.SUM,per_losses,axis=None)

@tf.function
def distributed_test_step(data_inputs):
    return mystrategy.run(test_step, args=(data_inputs,))

for epoch in range(25):
    whole_loss = 0.0
    num_batches = 0
    for x in dbtrain:
        whole_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = whole_loss / num_batches
    for x in dbval:
        distributed_test_step(x)
    print("epoch"+str(epoch+1)+": train_loss:"+str(float(train_loss))+
          " train_acc:"+str(float(train_acc.result()*100))+" val_loss:"
          + str(float(val_loss.result()))+" val_acc:"+str(float(val_acc.result()*100)))
    model.save_weights('./weights/epoch'+str(epoch+1)+'.h5',overwrite=True)
    # 设置提前终止策略
    loss_list.append(float(val_loss.result()))
    accuracy_list.append(float(val_acc.result()*100))
    max_accuracy_list.append(int(max(accuracy_list)))
    if max_accuracy_list.count(max_accuracy_list[-1]) > 15 :
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()
        break
    val_loss.reset_states()
    train_acc.reset_states()
    val_acc.reset_states()

with open('output.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(loss_list, accuracy_list,max_accuracy_list))






