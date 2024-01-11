import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from helpers.loss_functions import dice_coeff
import wandb
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.utils.clip_grad import clip_grad_value_

def plot_images(sample, outputs, epoch: int, i: int, phase: str):
    fig, ax = plt.subplot_mosaic([['true', 'pred'], ['img', 'img'], ['img', 'img']], layout='constrained')
    img_shape = sample[1][0].shape
    ax['true'].imshow(sample[1][0].reshape(img_shape))
    ax['pred'].imshow(outputs.data.cpu().numpy()[0].reshape(img_shape))
    ax['img'].imshow(np.transpose(sample[0][0], (1, 2, 0)))
    fig.savefig(f"custom_model/eval_imgs/{phase}/{epoch}_{i}")
    plt.clf()
    plt.cla()
    fig.clear()
    plt.close()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_and_test(
    model: nn.Module,
    dataloaders: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    w_b: bool = False,
    scheduler: LRScheduler = None,
    num_epochs=100,
    show_images=False,
    max_norm=1
    ):

    plotted_train_sample = False
    since = time.time()
    best_loss=1e10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = ['epoch', 'training_loss', 'test_loss', 'train_dice_coeff', 'val_dice_coeff']
    train_epoch_losses = []
    val_epoch_losses = []
    for epoch in range(1,num_epochs+1):
        i = 0
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        batchsummary = {a: [0] for a in fieldnames}
        batch_train_loss= 0.0
        batch_test_loss = 0.0
        
        for phase in ['train','val']:
            if phase =='train':
                model.train()
            else:
                model.eval()
            for sample in iter(dataloaders[phase]):
                inputs = sample[0].to(device)
                masks = sample[1].to(device)
                
                #masks = masks.unsqueeze(1)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    print(f"Input before forward: {inputs.shape}")
                    outputs = model(inputs)
                    print(outputs.shape)
                    print(sample[1].shape)
                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = sample[1].numpy().ravel()

                    batchsummary[f'{phase}_dice_coeff'].append(dice_coeff(y_pred, y_true))

                    if phase == 'train':
    

                        loss.backward()
                        #clip_grad_value_(model.parameters(), clip_value=max_norm)
                        optimizer.step()

                        batch_train_loss += loss.item() * sample[0].size(0)
                        if show_images and not plotted_train_sample:
                            plot_images(sample, outputs, epoch, i, phase)
                            plotted_train_sample = True


                    else:
                        # Show plot of preds and plot of true
                        if show_images:
                            plot_images(sample, outputs, epoch, i, phase)
                            i += 1
                            plotted_train_sample = False
                        batch_test_loss += loss.item() * sample[0].size(0)

            if phase == 'train':
                epoch_train_loss = batch_train_loss / len(dataloaders['train'])
                train_epoch_losses.append(epoch_train_loss)
            else:
                epoch_test_loss = batch_test_loss / len(dataloaders['val'])
                val_epoch_losses.append(epoch_test_loss)

            batchsummary['epoch'] = epoch
            
            print('{} Loss: {:.4f}'.format(phase, loss))
            if w_b:
                wandb.log({"epoch": epoch, f"{phase}_loss": loss})

        if scheduler is not None:
            scheduler.step()
        best_loss = np.max(batchsummary['val_dice_coeff'])
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        if w_b:
            wandb.log({"train_dice_coeff": batchsummary["train_dice_coeff"], "val_dice_coeff": batchsummary["val_dice_coeff"]})
        print(
            f'\t\t\t train_dice_coeff: {batchsummary["train_dice_coeff"]}, val_dice_coeff: {batchsummary["val_dice_coeff"]}')

    print('Best dice coefficient: {:4f}'.format(best_loss))

    return model, train_epoch_losses, val_epoch_losses