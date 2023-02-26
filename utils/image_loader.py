import tensorflow as tf
from tensorflow.python.data import AUTOTUNE

def load_image(file_path):
    return tf.io.decode_png(tf.io.read_file(file_path), channels=3)


def get_seg_filename(file_path):
    return tf.strings.regex_replace(file_path, "_orig.png", "_seg.png")


def load_pair(file_path):
    return load_image(file_path), load_image(get_seg_filename(file_path))


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