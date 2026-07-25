from io import BytesIO
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (255 * image).astype(np.uint8)

        self.writer.add_image(
            tag,
            image,
            global_step=step,
            dataformats="HWC"
        )
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return

        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (255 * img).astype(np.uint8)

            self.writer.add_image(
                f"{tag}/{i}",
                img,
                global_step=step,
                dataformats="HWC"
            )

        self.writer.flush()