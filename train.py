from utils.train_utils import read_config_file, get_callbacks, get_model
from utils.semseg_dataset import SemSegDataSet
from datetime import datetime
import glob


def main():
    # read train config
    conf = read_config_file("configs/train_config.json")

    model_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name_prefix = str(conf["model_prefix"])

    # get dataset
    train_data_path = conf["path_to_dataset"] + "train/"
    valid_data_path = conf["path_to_dataset"] + "valid/"

    train_imgs = glob.glob(train_data_path + "images/*")
    train_masks = glob.glob(train_data_path + "masks/*")

    valid_imgs = glob.glob(valid_data_path + "images/*")
    valid_masks = glob.glob(valid_data_path + "masks/*")

    train_data = SemSegDataSet(img_paths=train_imgs,
                               mask_paths=train_masks,
                               img_size=(conf["img_size"], conf["img_size"]),
                               channels=(3, 1),
                               crop_percent_range=(0.75, 0.95),
                               seed=42
                               )

    valid_data = SemSegDataSet(img_paths=valid_imgs,
                               mask_paths=valid_masks,
                               img_size=(conf["img_size"], conf["img_size"]),
                               channels=(3, 1),
                               crop_percent_range=(0.75, 0.95),
                               seed=42
                               )

    train_size = train_data.size
    valid_size = valid_data.size

    train_data = train_data.batch(batch_size=conf["batch_size"],
                                  shuffle=False,
                                  shuffle_buffer=train_data.size)

    valid_data = valid_data.batch(batch_size=conf["batch_size"])

    # get model
    model = get_model(conf["img_size"])
    model.summary()

    # get train callback functions
    callbacks = get_callbacks(conf, model_suffix)

    # train model
    model.fit(train_data,
              epochs=conf["num_epochs"],
              steps_per_epoch=train_size//conf["batch_size"],
              validation_data=valid_data if valid_data is not None else None,
              validation_steps=valid_size//conf["batch_size"] if valid_data is not None else None,
              callbacks=callbacks)

    # save model
    model.save(model_name_prefix + "_" + model_suffix + '.h5')


if __name__ == '__main__':
    main()
