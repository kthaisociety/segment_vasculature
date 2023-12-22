import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_and_test(model: nn.Module, dataloaders: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, num_epochs=100, show_images=False):
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
                
                masks = masks.unsqueeze(1)
                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, masks)

                    y_pred = outputs.data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()

                    batchsummary[f'{phase}_dice_coeff'].append(dice_coeff(y_pred, y_true))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        batch_train_loss += loss.item() * sample[0].size(0)

                    else:
                        # Show plot of preds and plot of true
                        if show_images:
                            fig, ax = plt.subplot_mosaic([['true', 'pred'], ['img', 'img'], ['img', 'img']], layout='constrained')
                            img_shape = sample[1][0].shape
                            ax['true'].imshow(sample[1][0].reshape(img_shape))
                            ax['pred'].imshow(outputs.data.cpu().numpy()[0].reshape(img_shape))
                            ax['img'].imshow(np.transpose(sample[0][0], (1, 2, 0)))
                            fig.savefig(f"eval_imgs/eval_{epoch}_{i}")
                            plt.clf()
                            plt.cla()
                            fig.clear()
                            plt.close()
                            i += 1
                        batch_test_loss += loss.item() * sample[0].size(0)

            if phase == 'train':
                epoch_train_loss = batch_train_loss / len(dataloaders['train'])
                train_epoch_losses.append(epoch_train_loss)
            else:
                epoch_test_loss = batch_test_loss / len(dataloaders['val'])
                val_epoch_losses.append(epoch_test_loss)

            batchsummary['epoch'] = epoch
            
            print('{} Loss: {:.4f}'.format(phase, loss))

        best_loss = np.max(batchsummary['val_dice_coeff'])
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(
            f'\t\t\t train_dice_coeff: {batchsummary["train_dice_coeff"]}, val_dice_coeff: {batchsummary["val_dice_coeff"]}')

    print('Best dice coefficient: {:4f}'.format(best_loss))

    return model, train_epoch_losses, val_epoch_losses