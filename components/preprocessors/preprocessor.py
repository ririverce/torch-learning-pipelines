import copy
import cv2


class Preprocessor:

    def __init__(self, augmentor_list, converter):
        self._augmentor_list = augmentor_list
        self._converter = converter

    def _load_image(self, batch_data):
        for data in batch_data:
            data["image"] = cv2.imread(data["image_path"])
        return batch_data

    def _augment(self, batch_data):
        for data in batch_data:
            for augmentor in self._augmentor_list:
                data = augmentor(data)
        return batch_data

    def _convert(self, batch_data):
        batch_data = self._converter.convert(batch_data)
        return batch_data

    def process(self, input_batch_data, no_augment=False):
        batch_data = copy.deepcopy(input_batch_data)
        batch_data = self._load_image(batch_data)
        if no_augment:
            batch_data = self._augment(batch_data)
        batch_data = [data for data in batch_data if len(data["labels"]) > 0]
        batch_data = self._convert(batch_data)
        return batch_data
