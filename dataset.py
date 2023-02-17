import glob
import os
from PIL import Image


def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Dataset:

    def __init__(self, image_root_path, data_transforms=None, image_format='jpg'):
        self.data_transforms = data_transforms
        self.image_root_path = image_root_path
        self.image_format = image_format
        self.images = []
        self.labels = []
        classes_folders = os.listdir(self.image_root_path)
        # print("classes_folders: ", classes_folders)
        for cls_folder in classes_folders:
            # folder_path = os.path.join(self.image_root_path, cls_folder)
            folder_path = self.image_root_path + "/" + cls_folder
            # print("folder_path: ", folder_path)
            # if os.path.isdir(folder_path):
            if os.path.isfile(folder_path):
                # images_path = os.path.join(folder_path, "*.{}".format(self.image_format))
                # print("images_path:", images_path)
                images = glob.glob(folder_path)
                # print("images:", images)
                self.images.extend(images)
                # print("images extended")
            else:
                print("이미지 추가 안됨")
        # print("self.images:", self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        label_name = os.path.basename(os.path.dirname(image_file))
        image = default_loader(image_file)
        if self.data_transforms is not None:
            image = self.data_transforms(image)

        # return image, int(label_name)
        return image, str(label_name)
