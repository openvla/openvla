"""Token OpenVLA preprocessing, independent of robot and benchmark utilities."""

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def crop_and_resize(image, crop_scale, batch_size):
    """Upstream TensorFlow center crop; preserve its bilinear 224px contract."""
    import tensorflow as tf

    if image.shape.ndims not in (3, 4):
        raise ValueError("center crop expects HWC or NHWC images")
    expanded = image.shape.ndims == 3
    if expanded:
        image = tf.expand_dims(image, axis=0)
    heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), (batch_size,))
    widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), (batch_size,))
    top, left = (1 - heights) / 2, (1 - widths) / 2
    boxes = tf.stack([top, left, top + heights, left + widths], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(batch_size), (224, 224))
    return image[0] if expanded else image
