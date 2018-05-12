# coding: utf-8
from __future__ import print_function

import os
import time

import tensorflow as tf

import model
import reader
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                                                   'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "风格模型")
tf.app.flags.DEFINE_string("image_file", "content.jpg", "输入图片")
tf.app.flags.DEFINE_string('target_file', 'res.jpg', '转换风格后的图片')

FLAGS = tf.app.flags.FLAGS


def main(_):
    #获取图片的长宽
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('图片尺寸为: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:

            #读取图片信息
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
            
            #增加维度
            image = tf.expand_dims(image, 0)

            generated = model.net(image, training=False)
            generated = tf.cast(generated, tf.uint8)

            #从tensor中删除所有大小是1的维度
            generated = tf.squeeze(generated, [0])

            #保存模型
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            """获取已训练好的model"""
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            """生成转换style后的image"""
            # 确认'generated'文件夹存在.
            generated_file = 'generated/res.jpg'
            if os.path.exists('generated') is False:
                os.makedirs('generated')

            # 将转换后的图片信息写入文件
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
