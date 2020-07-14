import tensorflow as tf
import random
AUTOTUNE = tf.data.experimental.AUTOTUNE


class SemSegDataSet(object):
    def __init__(self, img_paths, mask_paths, img_size=(224, 224),
                 channels=(3, 1), augmentation=False,  crop_percent_range=None, seed=None):
        self.img_paths = sorted(img_paths)
        self.mask_paths = sorted(mask_paths)
        self.img_size = img_size
        self.channels = channels
        self.augmentation = augmentation
        if crop_percent_range is not None:
            self.crop_percent_range = crop_percent_range
        else:
            self.crop_percent_range = None

        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000)
        self.size = len(self.img_paths) if not self.augmentation else 2 * len(self.img_paths)

    def resize_data_point(self, img, mask):
        img = tf.image.resize(img, self.img_size)
        mask = tf.image.resize(mask, self.img_size)

        return img, mask

    def normalize_data_point(self, img, mask):
        img = tf.cast(img, tf.float32) / 255.0
        mask = tf.cast(mask, tf.float32) / 255.0

        return img, mask

    def random_crop(self, img, mask):
        if self.crop_percent_range is not None:
            crop_percent = tf.cast(tf.random.uniform([],
                                                     minval=self.crop_percent_range[0],
                                                     maxval=self.crop_percent_range[1],
                                                     dtype=tf.float32,
                                                     seed=self.seed), tf.float32)

            crop_conditional = tf.cast(tf.random.uniform(
                [], maxval=2, dtype=tf.int32, seed=self.seed), tf.bool)

            img_shape = tf.cast(tf.shape(img), tf.float32)
            mask_shape = tf.cast(tf.shape(mask), tf.float32)

            img_h = tf.cast(img_shape[0] * crop_percent, tf.int32)
            img_w = tf.cast(img_shape[1] * crop_percent, tf.int32)

            mask_h = tf.cast(mask_shape[0] * crop_percent, tf.float32)
            mask_w = tf.cast(mask_shape[1] * crop_percent, tf.float32)

            img = tf.cond(crop_conditional, lambda: tf.image.random_crop(
                img, [img_h, img_w, self.channels[0]], seed=self.seed), lambda: tf.identity(img))
            mask = tf.cond(crop_conditional, lambda: tf.image.random_crop(
                mask, [mask_h, mask_w, self.channels[1]], seed=self.seed), lambda: tf.identity(mask))

        return img, mask

    def random_horizontal_flip(self, img, mask):
        combined = tf.concat([img, mask], axis=2)
        combined = tf.image.random_flip_left_right(combined, seed=self.seed)
        img, mask = tf.split(combined, self.channels, axis=2)

        return img, mask

    def read_data(self, path_img, path_mask):
        raw_img = tf.io.read_file(path_img)
        raw_mask = tf.io.read_file(path_mask)

        img = tf.image.decode_jpeg(raw_img, channels=self.channels[0])
        mask = tf.image.decode_jpeg(raw_mask, channels=self.channels[1])

        return img, mask

    def batch(self, batch_size, shuffle=False, shuffle_buffer=None):

        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.mask_paths))
        data = data.map(self.read_data, num_parallel_calls=AUTOTUNE)
        data = data.map(self.resize_data_point, num_parallel_calls=AUTOTUNE)

        if self.augmentation:
            augmented_data = data.map(self.random_crop, num_parallel_calls=AUTOTUNE)
            augmented_data = augmented_data.map(self.random_horizontal_flip, num_parallel_calls=AUTOTUNE)
            augmented_data = augmented_data.map(self.resize_data_point, num_parallel_calls=AUTOTUNE)

            data = data.concatenate(augmented_data)

        data = data.map(self.normalize_data_point, num_parallel_calls=AUTOTUNE)
        data = data.repeat()

        if shuffle:
            buffer = shuffle_buffer if shuffle_buffer is not None else self.size

            data = data.prefetch(AUTOTUNE)\
                       .shuffle(buffer)\
                       .batch(batch_size)
        else:
            data = data.batch(batch_size)\
                       .prefetch(AUTOTUNE)

        return data


