import os
import time
import datetime
from math import ceil, sqrt
import numpy as np
from scipy.misc import imsave
from six.moves import range
from six import iteritems
import tensorflow as tf

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
filter = k[ : , : , None , None ] / k.sum() * np.eye(3, dtype = np.float32)
channels = 1
config = {"N" : 8,"norm" : 1e-7,"filter" : filter,"MAX_IMAGES" : 1,"STEP_SIZE" : 1.0,"NUM_ITERATION" : 50 ,"MAX_FEATUREMAP" : 1024,"FORCE_COMPUTE" : False,"NUM_LAPLACIAN_LEVEL" : 4,"REGULARIZATION_STRENGTH" : 1e-3}
def get_config():
    return config
def parse_tensors_dict(graph, layer_name, value_feed_dict):
    x = []
    feed_dict = {}
    with graph.as_default() as g:
        op = get_operation(graph = g, name = layer_name)
        op_tensor = op.outputs[0]
        tensor_shape = op_tensor.get_shape().as_list()
        if not config["FORCE_COMPUTE"] and tensor_shape[-1] > config["MAX_FEATUREMAP"]:
            print("Skipping. Too many featuremaps. May cause memory errors.")
            return None
        for key_op, value in iteritems(value_feed_dict):
            tmp = get_tensor(graph = g, name = key_op.name)
            feed_dict[tmp] = value
            x.append(tmp)
        X_in = x[0]
        feed_dict[X_in] = feed_dict[X_in][:config["MAX_IMAGES"]]
    return op_tensor, x, X_in, feed_dict
def _write_deconv(images, layer, path_outdir):
    is_success = True
    images = _im_normlize(images)
    grid_images = _images_to_grid(images)
    path_out = os.path.join(path_outdir, layer.lower().replace("/", "_"))
    for i in range(len(grid_images)):
        time_stamp = time.time()
        grid_image_path = os.path.join(path_out, "deconvolution")
        is_success = make_dir(grid_image_path)
        if grid_images[i].shape[-1] == 1:
            imsave(os.path.join( grid_image_path, "grid_image.png"), grid_images[i][0,:,:,0], format = "png")
        else:
            imsave(os.path.join(grid_image_path, "grid_image.png"), grid_images[i][0], format = "png")
    return is_success
def write_results(results, layer, path_outdir, method):
    is_success = True
    if method == "deconv":
        is_success = _write_deconv(results, layer, path_outdir)
    return is_success
# if dir not exits make one
def _is_dir_exist(path):
    return os.path.exists(path)
def make_dir(path):
    is_success = True
    if not _is_dir_exist(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            is_success = False
    return is_success
def get_operation(graph, name):
    return graph.get_operation_by_name(name = name)
def get_tensor(graph, name):
    return graph.get_tensor_by_name(name = name)
# image or images normalization
def image_normalization(image, s = 0.1, ubound = 255.0):
    img_min = np.min(image)
    img_max = np.max(image)
    return (((image - img_min) * ubound) / (img_max - img_min + config["norm"])).astype('uint8')
def _im_normlize(images, ubound = 255.0):
    N = len(images)
    H, W, C = images[0][0].shape
    for i in range(N):
        for j in range(images[i].shape[0]):
            images[i][j] = image_normalization(images[i][j], ubound = ubound)
    return images
def convert_into_grid(Xs, ubound=255.0, padding=1):
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid.astype('uint8')
def _images_to_grid(images):
    grid_images = []
    if len(images) > 0:
        N = len(images)
        H, W, C = images[0][0].shape
        for j in range(len(images[0])):
            tmp = np.zeros((N, H, W, C))
            for i in range(N):
                tmp[i] = images[i][j]
            grid_images.append(np.expand_dims(convert_into_grid(tmp), axis = 0))
    return grid_images
def _lap_split(img):
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, config["filters"], [1, 2, 2, 1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, config["filters"] * 4, tf.shape(img), [1, 2, 2, 1])
        hi = img-lo2
    return lo, hi
def _lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = _lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]
def _lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, config["filters"]*4, tf.shape(hi), [1,2,2,1]) + hi
    return img
def _normalize_std(img):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img), axis = (1, 2, 3), keep_dims=True))
        return img/tf.maximum(std, config["EPS"])
def lap_normalize(img, channels, scale_n):
    K5X5 = k[ : , : , None , None ] / k.sum() * np.eye(channels, dtype = np.float32)
    config["K5X5"] = K5X5
    tlevels = _lap_split_n(img, scale_n)
    tlevels = list(map(_normalize_std, tlevels))
    out = _lap_merge(tlevels)
    return out
