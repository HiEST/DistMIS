import argparse
import json
import time
import cv2
import imageio
import numpy as np
import nibabel as nib
from tqdm import tqdm
from datetime import timedelta, datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import UNet3D, print_model
import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
import os

########################################################################################################################
#                                       DATA/DATASET UTILS                                                             #
########################################################################################################################


def labeled_image(image, label):
    """
    Merge image and label and return a labeled_image for 1 label class.
    @param image: Numpy Array with shape (W, H, D, C), C = number of channels
    @param label: Numpy Array with shape (W, H, D, 1)
    """
    image = cv2.standardize(image[:, :, :, 0], None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F).astype(np.uint8)
    image = np.expand_dims(image, -1)
    labeled_image = np.zeros_like(label[:, :, :, :])
    labeled_image[:, :, :, :] = image * (1 - label[:, :, :, :]) # remove tumor part from image
    labeled_image += label[:, :, :, :] * 255 # color labels

    return labeled_image


def visualize_gif(image, name='vis'):
    """
    Save a gif image for the labeled image. Modified for unet ouputs.
    @param image: Numpy Array with shape (W, H, D, 1)
    """
    image = image.squeeze()
    images = []
    for i in range(image.shape[0]):
        x = image[min(i, image.shape[0] - 1), :, :]
        y = image[:, min(i, image.shape[1] - 1), :]
        z = image[:, :, min(i, image.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave(f"./gif_{name}.gif", images, duration=0.1)


def visualize_sample(train_dataloader):
    """
    Take an slice of train_dataloader (1 batch_size), that is a sample, and visualize it
    with saving a gif on your current directory.
    @param train_dataloader: DataLoder with batch size 1. Shape = ((1, C, W, H, D), (1, 1, W, H, D))
    """
    img, lbl = next(iter(train_dataloader))
    img = np.moveaxis(img.squeeze().numpy(), 0, -1)
    lbl = np.moveaxis(lbl.squeeze(axis=0).numpy(), 0, -1)
    print(img.shape, lbl.shape)
    visualize_gif(labeled_image(img, lbl))


def standardize(a, axis=None):
    """
    Standardize image a along axis.
    @param a: Numpy Array
    @param axis: axis along which mean & std reductions are performed
    """
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std


class BrainMRIDataset(Dataset):
    """
    Dataset Class for Brain MRI data located on base_dir
    """
    def __init__(self, base_dir, transform=None, target_transform=None):
        with open(base_dir + 'dataset.json') as f:
            dataset = json.load(f)
        self.base_dir = base_dir
        self.data = [(d['image'], d['label']) for d in dataset['training']]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_dir, label_dir = self.data[idx]
        image = np.array(nib.load(self.base_dir + image_dir[1:]).get_fdata(), dtype=np.float32)[:, :, 2:-1, :]
        label = np.array(nib.load(self.base_dir + label_dir[1:]).get_fdata(), dtype=np.float32)[:, :, 2:-1]
        y = np.zeros(label.shape)
        y[(label > 0) & (label < 4)] = 1
        image = np.moveaxis(image, -1, 0)
        image = standardize(image, axis=(1, 2, 3))
        label = np.expand_dims(y, 0)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


########################################################################################################################
#                                       MAIN CODE                                                                      #
########################################################################################################################

def dice_coeff(pred, target, smooth=1):
    """
    Sorensen-Dice coefficient (DSC) metric
    @param pred: torch.tensor, model prediction
    @param target: torch.tensor, ground truth label
    @param smooth: Laplace smooth, a large value can be used to avoid overfitting
    """
    iflat, tflat = pred.view(-1), target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def dice_loss(pred, target):
    """
    Dice Loss based on Dice Similarity Coefficient (DSC)
    @param pred: torch.tensor, model prediction
    @param target: torch.tensor, ground truth label
    """
    return 1 - dice_coeff(pred, target)


def train_loop(dataloader, model, loss_fn, metric, optimizer, **kwargs):
    """
    Training loop for fit function
    """
    pbar = dataloader
    if kwargs['verbose'] == 1:
        pbar = tqdm(dataloader, ncols=90, unit='step')
        pbar.set_description(f"Epoch {kwargs['epoch']}")

    for _, (x, y) in enumerate(pbar, 1):
        x, y = x.to(kwargs['device']), y.to(kwargs['device'])  # send tensors to VRAM

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(x)
        loss, acc = loss_fn(pred, y), metric(pred, y)
        loss.backward()
        optimizer.step()

        if kwargs['verbose'] == 1:
            pbar.set_postfix(loss=loss.item(), acc=acc.item())

    return {'loss': loss.item(), 'acc': acc.item()}


def test_loop(dataloader, model, loss_fn, metric, **kwargs):
    """
    Test loop for both fit and evaluate functions
    """
    size, val_loss, val_acc = len(dataloader), 0, 0
    pbar = dataloader

    if kwargs['verbose'] == 1:
        pbar = tqdm(dataloader, ncols=90, unit='step')
        pbar.set_description(f"Epoch {kwargs['epoch']}" if 'epoch' in kwargs else "Evaluation")
    elif kwargs['verbose'] == 2:
        _s = time.time()

    with torch.no_grad():
        for i, (x, y) in enumerate(pbar, 1):
            x, y = x.to(kwargs['device']), y.to(kwargs['device'])  # send tensors to VRAM
            pred = model(x)
            val_loss += loss_fn(pred, y).item()
            val_acc += metric(pred, y).item()
            if kwargs['verbose'] == 1:
                pbar.set_postfix(val_loss=val_loss / i, val_acc=val_acc / i)
    val_loss /= size
    val_acc /= size

    return {'loss': val_loss, 'acc': val_acc}


def fit(model, train_dataloader, valid_dataloader, epochs, loss_fn, metric, opt, device='cuda',
        tb_logdir="", verbose=1):
    """
    Training of a model on the train_dataloader set for epochs and validation in each epoch
    @param model: nn.Module class which represents the model
    @param train_dataloader: data.DataLoader class which represents the train set
    @param valid_dataloader: data.DataLoader class which represents the validation set
    @param epochs: Number of epochs to train the model.
    @param loss_fn: Function loss to train and validate the model
    @param metric: Function metric to validate the model
    @param opt: Optimizer used to train the model
    @param device: Device where the model is executed. Default 'cuda'.
    @param tb_logdir: Dir where tensorboard will save its logs.
    @param verbose: Verbosity outputs, 0 = silence, 1 = every step, 2 = every epoch
    """
    tstamp = "{}".format(datetime.now().strftime("%m_%d_%Y-%H:%M:%S"))
    s_t = time.time()
    for epoch in range(1, epochs+1):
        if verbose == 2:
            _s = time.time()

        train_logs = train_loop(train_dataloader, model, loss_fn, metric, opt,
                                epoch=epoch, verbose=verbose, device=device)

        valid_logs = test_loop(valid_dataloader, model, loss_fn, metric, epoch=epoch,
                               verbose=verbose, device=device)
        if tb_logdir:
            with SummaryWriter(f'{tb_logdir}/logs/{tstamp}/train/') as writer:
                writer.add_scalar('loss', train_logs['loss'], epoch)
                writer.add_scalar('dice_coefficient', train_logs['acc'], epoch)

            with SummaryWriter(f'{tb_logdir}/logs/{tstamp}/validation/') as writer:
                writer.add_scalar('loss', valid_logs['loss'], epoch)
                writer.add_scalar('dice_coefficient', valid_logs['acc'], epoch)

        if verbose == 2:
            m, s = divmod(time.time() - _s, 60)
            print(f"Epoch {epoch}: 100% | [{int(m):0>2}:{int(s):0>2}, "
                  f"loss={train_logs['loss']:.4f}, acc={train_logs['acc']:.4f},"
                  f" val_loss={valid_logs['loss']:.4f}, val_acc={valid_logs['acc']:.4f}]")

    print("Elapsed training time:", timedelta(seconds=(time.time() - s_t)), "s")


def evaluate(model, test_dataloader, loss_fn, metric, device='cuda', verbose=1):
    """
    Evaluation of a model on the test_dataloader set using loss_fn and metric.
    @param model: nn.Module class which represents the model
    @param test_dataloader: data.DataLoader class
    @param loss_fn: Function loss to evaluate the model
    @param metric: Function metric to evaluate the model
    @param device: Device where the model is executed. Default 'cuda'.
    @param verbose: Verbosity outputs, 0 = silence, 1 = every step, 2 = every epoch
    """
    _s = time.time()
    test_logs = test_loop(test_dataloader, model, loss_fn, metric, device=device, verbose=verbose)
    if verbose == 2:
        m, s = divmod(time.time() - _s, 60)
        print(f"Evaluation : 100% | [{int(m):0>2}:{int(s):0>2}, "
              f"loss={test_logs['loss']:.4f}, acc={test_logs['acc']:.4f}]")


def model_creator(config):
    return UNet3D()


def data_creator(args):
    # Create dataset
    base_path = "/gpfs/projects/bsc31/bsc31654/dataset/"
    dataset = BrainMRIDataset(base_path)

    # Split dataset into train, valid, test sets
    train_len, valid_len = [int(p * len(dataset)) for p in (0.7, 0.15)]
    test_len = len(dataset) - train_len - valid_len  # test ~ 0.15
    train, valid, test = random_split(dataset, [train_len, valid_len, test_len])

    # Create DataLoaders
    train_dataloader = DataLoader(train, batch_size=args.get('batch_size'), shuffle=True, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(valid, batch_size=args.get('batch_size'), shuffle=True, num_workers=8, pin_memory=True)

    return train_dataloader, valid_dataloader


def loss_creator(config):
    return dice_loss


def optimizer_creator(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.get('lr'))


def train_example(args):
    """
    Defines a TorchTrainer object that will distribute the data among all the available gpus.
    @param args: arguments that define the training of the model
    """
    CustomTrainingOperator = TrainingOperator.from_creators(
        model_creator=model_creator, optimizer_creator=optimizer_creator,
        data_creator=data_creator, loss_creator=loss_creator)

    trainer1 = TorchTrainer(
        training_operator_cls=CustomTrainingOperator,
        config=vars(args),
        num_workers=args.num_workers,
        use_gpu=args.use_gpu,
        backend="nccl",
        scheduler_step_freq="epoch"
    )

    s_t = time.time()
    for epoch in range(1, args.epochs + 1):
        print('Epoch '+str(epoch))
        if args.verbose == 2:
            _s = time.time()

        stats_train = trainer1.train()
        stats_valid = trainer1.validate()

        if args.verbose == 2:
            m, s = divmod(time.time() - _s, 60)
            print(f"Epoch {epoch}: 100% | [{int(m):0>2}:{int(s):0>2}, "
                  f"loss={stats_train['train_loss']:.4f},"f" val_loss={stats_valid['val_loss']:.4f}, "
                  f"val_acc={stats_valid['val_accuracy']:.4f}]")

    print("Elapsed training time:", timedelta(seconds=(time.time() - s_t)), "s")
    trainer1.shutdown()
    print("success!")


def main(gpus):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, help="json file configuration")
    args, _ = parser.parse_known_args()

    with open('../configs/'+str(gpus)+'gpu/config.json') as f:
        config = json.load(f)

    args.epochs = config.get('epochs', 200)
    args.verbose = config.get('verbose', 2)
    args.use_gpu = config.get('use_gpu', False)
    args.num_workers = config.get('num_workers')
    args.adress = config.get('adress')
    args.lr = config.get('lr')
    args.batch_size = config.get('batch_size', 2)

    train_example(args)


if __name__ == '__main__':
    main()
