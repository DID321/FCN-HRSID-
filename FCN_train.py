import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, callbacks
from matplotlib import pyplot as plt
from FCN import FCN_Net
import pathlib
import datetime
from PIL import Image
import os

def read_image(image_path,image_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32) / 255.0  # 归一化到[0,1]范围

    mask = tf.io.read_file(image_label)
    mask = tf.image.decode_jpeg(mask,channels=1)
    return image, mask

def get_data(voc_root, txt_name):

    root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
    image_dir = os.path.join(root, 'JPEGImages')
    mask_dir = os.path.join(root, 'SegmentationClassPmode')

    txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
    with open(os.path.join(txt_path), "r") as f:
        file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

    images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
    masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

    print(txt_name, " images: ",len(images))
    print(txt_name, " masks: ", len(masks))


    db = tf.data.Dataset.from_tensor_slices((images,masks))
    db = db.map(read_image)
    db = db.batch(4)

    return db

def main():    
    '''main函数'''
    model = FCN_Net()
    model.summary()

    train_db = get_data("./", "train.txt")
    test_db = get_data("./", "val.txt")

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/FCNNets_epoch25_' + current_time
    # tb_callback = callbacks.TensorBoard(log_dir=log_dir)

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #              metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss=tf.compat.v2.nn.softmax_cross_entropy_with_logits,
                  metrics=['accuracy'])
    model.fit(train_db, epochs=25, validation_data=test_db, validation_freq=1)
    model.evaluate(test_db)

    model.save_weights('./FCN_epoch25_weights' + current_time + '.ckpt')
    print('save weights')
    '''
    model.fit(train_db, epochs=50, validation_data=test_db,validation_freq=1,callbacks=[tb_callback])
    #model.load_weights('./checkpoint/AConvNets_SOC_epoch50_weights.ckpt') 

    model.evaluate(test_db)

    model.save_weights('./checkpoint/AConvNets_ATR_epoch50_weights.ckpt')
    print('save weights')
    '''

if __name__ == '__main__':
    main()

    