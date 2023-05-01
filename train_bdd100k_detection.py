import components
import numpy as np
import cv2
import torch
import tqdm


def main():
    loader = components.loaders.BDD100KDetectionLoader("/home/hal/dataset/berkeley_deep_drive/bdd100k")
    preprocessor = components.preprocessors.Preprocessor(
        augmentor_list=[
            components.preprocessors.augmentors.detection_augmentor.bbox_random_crop
        ],
        converter=components.preprocessors.converters.AnchorBoxConverter(
            11, components.utils.anchor_box.default_box_ssdwide(), (400, 225)
        )
    )

    """ device """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    """ model """
    model = components.models.SSDWideLiteVGG16(input_channels=3,
                                              num_classes=11,
                                              num_bboxes=[4, 6, 6, 6, 4, 4]).to(device)

    """ loss """
    loss_function = components.losses.MultiBoxLoss().to('cpu')

    """ optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                       T_max=20)

    """ loop """
    for epoch in range(1, 100):
        train_dataset = loader.get_train_dataset()
        batch_size = 32
        train_batch_list = [
            train_dataset[i:i+batch_size] for i in range(0, len(train_dataset), batch_size)
        ]
        print(f"-"*64)
        print(f"[epoch {epoch:>4d}]")
        loss_list = []
        torch.set_grad_enabled(True)
        for batch_data in tqdm.tqdm(train_batch_list):
            batch_data = preprocessor.process(batch_data)
            optimizer.zero_grad()
            batch_image = torch.from_numpy(batch_data['input']).to(device)
            batch_gt_conf = torch.from_numpy(batch_data['conf']).to('cpu')
            batch_gt_loc = torch.from_numpy(batch_data['loc']).to('cpu')
            batch_output = model(batch_image)
            batch_pred_conf, batch_pred_loc = batch_output
            batch_pred_conf = batch_pred_conf.to('cpu')
            batch_pred_loc = batch_pred_loc.to('cpu')
            batch_loss_conf, batch_loss_loc = loss_function(batch_pred_conf,
                                                            batch_pred_loc,
                                                            batch_gt_conf,
                                                            batch_gt_loc)
            batch_loss = batch_loss_conf + batch_loss_loc
            batch_loss = batch_loss.sum() / batch_loss.shape[0]
            batch_loss.backward()
            optimizer.step()
            batch_loss = (batch_loss_conf + batch_loss_loc).detach().cpu().numpy()
            loss_list.extend(batch_loss)
            print(np.mean(loss_list))


if __name__ == "__main__":
    main()