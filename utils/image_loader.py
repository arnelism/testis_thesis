import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


def load_image(file_path):
    image = tf.io.decode_png(tf.io.read_file(file_path), channels=3)
    # normalize to 0...1 range
    return tf.cast(image, tf.float32)/255


def get_seg_filename(file_path):
    return tf.strings.regex_replace(file_path, "_orig.png", "_seg.png")


def load_seg_image(file_path):
    image = tf.io.decode_png(tf.io.read_file(get_seg_filename(file_path)), channels=4)

    # rewrite blue channel based on alpha channel
    # max blue when alpha==0(transparent img)
    # also normalize to 0...1 range (xxx why did I use green=128 when generating train data?)
    red, green, blue, alpha = tf.unstack(image, axis=-1)
    return tf.stack ([
        tf.cast(red, tf.float32)/255,
        tf.cast(green, tf.float32)/128,
        tf.cast(255-alpha, tf.float32)/255
    ], axis=-1)


def load_pair(file_path):
    return load_image(file_path), load_seg_image(file_path)


def get_image_loader(path: str, batch_size: int):
    image_files = tf.data.Dataset.list_files(
        f"{path}/*_orig.png", shuffle=True
    )
    images = image_files.map(load_pair, num_parallel_calls=AUTOTUNE)
    print(f"Creating dataset for {len(images)} image pairs")

    train_batches = (
        images
        .cache()
        .shuffle(1000)
        .batch(batch_size)
        # .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches