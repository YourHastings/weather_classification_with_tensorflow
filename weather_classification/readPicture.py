import pathlib
import random
import tensorflow as tf

def readpic(data_dir):
    data_root = pathlib.Path(data_dir)
    all_image_paths = list(data_root.glob('*/*')) # 所有子目录下图片
    all_image_paths = [str(path) for path in all_image_paths] # many a picture
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                            for path in all_image_paths]
    return all_image_paths,all_image_labels

def preprocess_image(path,image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])
    image = random_transform(image,image_size)
    image /= 255.0
    return image

def random_transform(image,image_size):
    type = random.randrange(1,11)
    if type < 7:
        return image
    elif type == 7:
        image = tf.image.resize(image, [50, 50])
        return tf.image.resize(image, [image_size, image_size])
    elif type == 8 :
        return tf.image.transpose(image)
    elif type == 9 :
        return tf.image.flip_up_down(image)
    else:
        image = tf.image.central_crop(image,central_fraction=0.6)
        return tf.image.resize(image, [image_size, image_size])



