import sys
import random
import components
import numpy as np
import cv2
import torch
import tqdm


class BDD100KDetectionTrainer:

    def __init__(self, dataset_dir, augmentor_list, converter, batch_size=32):
        self._batch_size = batch_size
        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._loader = components.loaders.BDD100KDetectionLoader(dataset_dir)
        self._preprocessor = components.preprocessors.Preprocessor(
            augmentor_list=augmentor_list, converter=converter)
        self._model = components.models.SSDWideLiteVGG16(
            input_channels=3, num_classes=11, num_bboxes=[4, 6, 6, 6, 4, 4]
        ).to(self._device)
        self._loss_function = components.losses.MultiBoxLoss().to('cpu')
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)

    def _train_one_batch(self, batch_data):
        batch_data = self._preprocessor.process(batch_data)            
        batch_image = torch.from_numpy(batch_data['input']).to(self._device)
        batch_gt_conf = torch.from_numpy(batch_data['conf']).to('cpu')
        batch_gt_loc = torch.from_numpy(batch_data['loc']).to('cpu')
        batch_output = self._model(batch_image)
        batch_pred_conf, batch_pred_loc = batch_output
        batch_pred_conf = batch_pred_conf.to('cpu')
        batch_pred_loc = batch_pred_loc.to('cpu')
        batch_loss = self._loss_function(batch_pred_conf, batch_pred_loc,
                                            batch_gt_conf, batch_gt_loc)
        self._optimizer.zero_grad()
        batch_loss.backward()
        self._optimizer.step()
        batch_loss.detach().numpy().copy()
        return batch_loss

    def _train_one_epoch(self):
        dataset = self._loader.get_train_dataset()
        random.shuffle(dataset)
        torch.set_grad_enabled(True)
        loss_list = []
        pbar = tqdm.tqdm(range(0, len(dataset), self._batch_size))
        for i in pbar:
            batch_data = dataset[i:i+self._batch_size]
            batch_loss = self._train_one_batch(batch_data)
            loss_list.append(batch_loss)
            pbar.set_description(
                f"[train total_loss={np.mean(loss_list):0.4f}]")
            pbar.set_postfix({"loss": loss_list[-1]})
        pbar.close()
        loss = np.mean(loss_list)
        return loss, 0

    def _inference_one_batch(self, batch_data):
        batch_data = self._preprocessor.process(batch_data, no_augment=True)
        batch_image = torch.from_numpy(batch_data['input']).to(self._device)
        batch_gt_conf = torch.from_numpy(batch_data['conf']).to('cpu')
        batch_gt_loc = torch.from_numpy(batch_data['loc']).to('cpu')
        batch_output = self._model(batch_image)
        batch_pred_conf, batch_pred_loc = batch_output
        batch_pred_conf = batch_pred_conf.to('cpu')
        batch_pred_loc = batch_pred_loc.to('cpu')
        batch_loss = self._loss_function(batch_pred_conf, batch_pred_loc,
                                         batch_gt_conf, batch_gt_loc)
        batch_loss = batch_loss.detach().numpy().copy()
        return batch_loss
    
    def _inference_one_epoch(self):
        dataset = self._loader.get_val_dataset()
        loss_list = []
        pbar = tqdm.tqdm(range(0, len(dataset), self._batch_size))
        for i in pbar:
            batch_data = dataset[i:i+self._batch_size]
            batch_loss = self._inference_one_batch(batch_data)
            loss_list.append(batch_loss)
            pbar.set_description(
                f"[validation total_loss={np.mean(loss_list):0.4f}]")
            pbar.set_postfix({"loss": loss_list[-1]})
        pbar.close()
        loss = np.mean(loss_list)
        return loss, 0

    def start(self, num_epoch=100):
        for epoch in range(1, num_epoch+1):
            t_loss, t_acc = self._train_one_epoch()
            v_loss, v_acc = self._inference_one_epoch()
            print(f"epoch {epoch:>4d}")
            print(f"train: loss={t_loss:0.4f}, acc={t_acc:0.2d}")
            print(f"validation: loss={v_loss:0.4f}, acc={v_acc:0.2d}")
        


def main():
    augmentor_list = [
        components.preprocessors.augmentors.detection_augmentor.bbox_random_crop
    ]
    converter = components.preprocessors.converters.AnchorBoxConverter(
        11, components.utils.anchor_box.default_box_ssdwide(), (400, 225))
    trainer = BDD100KDetectionTrainer(
        "/home/hal/dataset/berkeley_deep_drive/bdd100k/",
        augmentor_list=augmentor_list,
        converter=converter)
    trainer.start()


if __name__ == "__main__":
    main()