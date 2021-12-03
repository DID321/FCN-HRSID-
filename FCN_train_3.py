import tensorflow as tf 
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, callbacks
from FCN_2 import FCN_Net
import pathlib
import datetime
import os
from tqdm import tqdm
def read_image(image_path,image_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, dtype=tf.float32) / 127.5 -1 # 归一化到[-1,1]范围

    mask = tf.io.read_file(image_label)
    mask = tf.image.decode_png(mask,channels=1)
    mask = tf.one_hot(mask[:,:,0], 2)

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

    return db

def main():    
    '''main函数'''
    model = FCN_Net()
    model.summary()

    train_db = get_data("./FCN", "train.txt")
    test_db = get_data("./FCN", "val.txt")
    """
    for img, musk in train_db.take(1):
        plt.subplot(1, 2, 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[1]))
        plt.subplot(1, 2, 2)
        print(musk[1].numpy().max())
        mask_new = np.where(musk[1].numpy()>0,255,0)
        print(mask_new.max())
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask_new))

    plt.show()
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = 'logs/FCNNets_epoch25_' + current_time
    # tb_callback = callbacks.TensorBoard(log_dir=log_dir)

    # model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #              metrics=['accuracy'])
    Epoch = 15

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_acc")

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_acc = tf.keras.metrics.CategoricalAccuracy(name="test_acc")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    for epoch in range(Epoch):
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()
        print("epoch:", epoch+1, " train:")
        for step, (x, y) in tqdm(enumerate(train_db)):
            with tf.GradientTape() as tape:
                logits1 = model(x)
                loss = tf.nn.weighted_cross_entropy_with_logits(y, logits1, pos_weight=1.5)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_acc.update_state(y, logits1)
        print("epoch:", epoch + 1, " test:", end=" ")
        for x, y in tqdm(test_db):
            logits = model.predict(x)
            t_loss = tf.nn.weighted_cross_entropy_with_logits(y, logits, pos_weight=1.5)
            test_loss.update_state(t_loss)
            test_acc.update_state(y, logits)

        template = "Epoch: {}, Loss: {}, Acc: {}%, Test_Loss: {}, Test_Acc: {}%"
        print(template.format(epoch+1, train_loss.result(), train_acc.result()*100, test_loss.result(), test_acc.result()*100))
        model.save_weights('./FCN_with_focal_loss_epoch15_weights_epoch' + str(epoch+1) + '.ckpt')
        print('save weights',epoch+1)

    model.save_weights('./FCN_with_focal_loss_epoch15_weights' + current_time + '.ckpt')
    print('save weights')

if __name__ == '__main__':
    main()

    