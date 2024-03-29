from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
STYLE_LAYERS = ( 'relu5_2','relu5_3','relu5_4')
CONTENT_LAYER = 'relu2_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, class_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False, if_continue = 0, checkpoint_dir = 'checkpoint'):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod] 

    style_features = {}

    batch_shape = (batch_size,224,224,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)

    # precompute style features
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session(config=config) as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            if   layer == 'relu5_4' or layer == 'relu5_3' or layer == 'relu5_2' or layer == 'relu5_1' :
                style_features[layer] = net[layer].eval(feed_dict={style_image:style_pre})
            else:
                features = net[layer].eval(feed_dict={style_image:style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[layer] = gram

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random_normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            #X_content_336 = tf.image.resize_image_with_crop_or_pad(X_content,336,336)
            preds = transform.net(X_content/255.0)
            #preds_crop = tf.image.resize_image_with_crop_or_pad(preds,224,224)
            #preds_224 = tf.image.resize_images(preds,[224,224])
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        #content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        content_size = _tensor_size(preds) * batch_size
        #assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        #content_loss = content_weight * (2 * tf.nn.l2_loss(
        #     net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        # )
        content_loss2 = content_weight  * (2 * tf.nn.l2_loss(preds - X_content) / content_size)

        # class loss
        #preds_pre_224 = tf.image.resize_image_with_crop_or_pad(preds_pre, 224, 224)
        #net_class = vgg.net(vgg_path, preds_pre_224)
        class_loss = -tf.reduce_sum(net['prob'][:,327])*class_weight

        style_losses = []
        style_layer_weight = 1
        for style_layer in STYLE_LAYERS:
            if  style_layer == 'relu5_4' or style_layer == 'relu5_3' or style_layer == 'relu5_2' or style_layer == 'relu5_1':
                grams = net[style_layer]
            else:
                layer = net[style_layer]
                bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
                size = height * width * filters
                feats = tf.reshape(layer, (bs, height * width, filters))
                feats_T = tf.transpose(feats, perm=[0,2,1])
                grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(style_layer_weight * 2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
                #style_layer_weight = 0.2

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


        loss = style_loss + tv_loss + class_loss + content_loss2

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        if if_continue:
            model_file = tf.train.latest_checkpoint(checkpoint_dir)
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (224,224,3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("UID: %s, batch time: %s" % (uid, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss2, tv_loss, class_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_class_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss,_class_loss, _loss)


                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)

                    yield(_preds,X_batch, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
