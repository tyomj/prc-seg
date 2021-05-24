import cv2
import numpy as np
import torch

from src.pl_modules import InferenceModel

WEIGHTS_PATH = 'weights/price_segmentation.ckpt'
IN_IMAGE = 'test/input.jpg'
OUT_IMAGE = 'test/output.jpg'


def main() -> None:
    # model
    model = InferenceModel.load_from_checkpoint(WEIGHTS_PATH)
    model.eval()
    # image
    image = cv2.imread(IN_IMAGE, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w, _ = image.shape
    # inference
    with torch.no_grad():
        mask = model(image)[0][0].numpy()
    # plot
    resized_masks = cv2.resize(mask, (w, h))
    x1, y1, x2, y2 = model.mask2bbox(resized_masks)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.imwrite(OUT_IMAGE, image)


if __name__ == '__main__':
    main()
