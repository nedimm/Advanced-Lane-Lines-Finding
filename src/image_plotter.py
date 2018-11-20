import math
import matplotlib.pyplot as plt


class ImagePlotter:
    def __init__(self, images_to_show_count=6):
        self.images_count = images_to_show_count
        self.images = {}

    def add_to_plot(self, image, filename, category, show_gray):
        if category not in self.images:
            self.images[category] = []
        if len(self.images[category]) < self.images_count:
            self.images[category].append((image, filename, show_gray))

    def plot(self, category, cols):
        if category in self.images:
            self.show_images(self.images[category], cols)

    def show_images(self, images, cols):
        rows = math.ceil(len(images) / cols)
        fix, ax = plt.subplots(rows, cols, figsize=(10, 10))

        img_cnt = 0
        img_length = len(images)
        for i in range(rows):
            for j in range(cols):
                image, title, show_gray = images[img_cnt]
                if (rows > 1):
                    ax[i][j].set_title(title, fontsize=10)
                    if show_gray:
                        ax[i][j].imshow(image, cmap='gray')
                    else:
                        ax[i][j].imshow(image)
                else:
                    ax[j].set_title(title, fontsize=10)
                    if show_gray:
                        ax[j].imshow(image, cmap='gray')
                    else:
                        ax[j].imshow(image)
                img_cnt = img_cnt + 1
                if img_cnt >= img_length:
                    break
            if img_cnt >= img_length:
                break
        plt.tight_layout()
        plt.show()
