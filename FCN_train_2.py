import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, callbacks
from FCN_2 import FCN_Net
import pathlib
import datetime
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

def read_image(image_path,image_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, dtype=tf.float32) / 127.5 -1 # 归一化到[-1,1]范围

    mask = tf.io.read_file(image_label)
    mask = tf.image.decode_png(mask,channels=1)


    mask = tf.one_hot(mask[:,:,0], 2, dtype = tf.float32)

    return image, mask

def get_data(voc_root, txt_name):

    root = os.path.join(voc_root, "VOCdevkit", "VOC2012")
    image_dir = os.path.join(root, 'JPEGImages')
    mask_dir = os.path.join(root, 'SegmentationClassLmode')

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

    return db, len(images)

def weighted_loss(labels, logits):
    '''
    Weighted loss.
    Args:
    labels: without onehot
    logits: after sorfmax

    Return loss
    '''
    sf_logits_log = (-1) * tf.math.log(logits) #[N, c]
    num_class = logits.shape[-1]
    #oh_labels = tf.one_hot(labels, num_class, dtype = tf.float32) #[N, c]
    #set 1.2 for tumor and 0.8 for normal
    y_true = 5 * labels[:, 1:]
    y_false = 0.5 * labels[:, 0:1]
    weight_labels = tf.concat([y_false, y_true], axis = 1)
    loss = tf.reduce_sum(sf_logits_log * weight_labels, axis = 1)
    loss = tf.reduce_mean(loss)
    return loss

def main():    
    '''main函数'''
    model = FCN_Net()
    model.summary()

    train_db, train_nums = get_data("../FCN", "train.txt")
    test_db, test_nums = get_data("../FCN", "val.txt")
    """
    for img, musk in train_db.take(1):
        plt.subplot(1, 2, 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[1]))
        plt.subplot(1, 2, 2)
        print(musk[1,:,:,0].numpy().max())
        mask_new = np.zeros((800,800,1))
        mask_new[:,:,0] = np.where(musk[1,:,:,1].numpy()>0,255,0)
        print(mask_new.shape)
        print(mask_new.max())
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask_new))

    plt.show()
"""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/FCNNets_epoch25_' + current_time
    # tb_callback = callbacks.TensorBoard(log_dir=log_dir)

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #              metrics=['accuracy'])
    total_epoch = 5

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_acc = tf.keras.metrics.CategoricalAccuracy(name="test_acc")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    if train_nums % 4 == 0:
        train_batch_nums = int(train_nums / 4)
    else:
        train_batch_nums = int(train_nums / 4) + 1

    if test_nums % 4 == 0:
        test_batch_nums = int(test_nums / 4)
    else:
        test_batch_nums = int(test_nums / 4) + 1


    for epoch in range(total_epoch):
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        with tqdm(total=train_batch_nums) as pbar:
            pbar.set_description('epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀 一般为epoch的信息
            for step, (x, y) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    logits1 = model(x)
                    #loss = tf.nn.weighted_cross_entropy_with_logits(y, logits1, pos_weight=10)
                    loss = weighted_loss(y, logits1)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss.update_state(loss)
                train_acc.update_state(y, logits1)

                pbar.set_postfix(loss='{:.6f}'.format(train_loss.result()))
                pbar.update(1)
        with tqdm(total=test_batch_nums) as pbar:
            pbar.set_description('epoch: {}/{} Test:'.format(epoch + 1, total_epoch))  # 设置前缀 一般为epoch的信息
            for step, (x, y) in enumerate(test_db):
                logits = model.predict(x)
                #t_loss = tf.nn.weighted_cross_entropy_with_logits(y, logits, pos_weight=10)
                t_loss = weighted_loss(y, logits)
                test_loss.update_state(t_loss)
                test_acc.update_state(y, logits)
                pbar.set_postfix(loss='{:.6f}'.format(test_loss.result()))
                pbar.update(1)

        template = "Epoch: {}, Loss: {}, Acc: {}%, Test_Loss: {}, Test_Acc: {}%"
        print(template.format(epoch+1, train_loss.result(), train_acc.result()*100, test_loss.result(), test_acc.result()*100))

    model.save_weights('./FCN_with_focal_loss_epoch5_weights' + current_time + '.ckpt')
    print('save weights')

if __name__ == '__main__':
    main()

    