from copy import deepcopy
import time

import torch
from torch import optim
from torch.optim import lr_scheduler

from data_processing import get_dataloader
from metrics import *
from pytorch_models import *


def train_model(model, train_loader, valid_loader, optimizer, scheduler, num_epochs=25):
    # Choose loss function (from tutorial)
    criterion = nn.CrossEntropyLoss()

    dataloaders = {}
    dataloaders['train'] = train_loader
    dataloaders['val'] = valid_loader

    dataset_sizes = {}
    dataset_sizes['train'] = len(train_loader) * len(train_loader[0])
    dataset_sizes['val'] = len(valid_loader) * len(valid_loader[0])

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    img_size = (128, 128)
    batch_size = 64

    trn_lbl_path = "../Dataset/train.txt"
    trn_img_path = "../Dataset/train/"
    train_loader = get_dataloader(trn_img_path, trn_lbl_path, img_size, batch_size)

    valid_lbl_path = "../Dataset/test.txt"
    valid_img_path = "../Dataset/test/"
    valid_loader = get_dataloader(valid_img_path, valid_lbl_path, img_size, batch_size)

    custom_model = CustomCNN()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use gpu if avaialble (from tutorial)
    custom_model = custom_model.to(device)

    # Observe that all parameters are being optimized (from tutorial)
    optimizer_custom = optim.SGD(custom_model.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs (from tutorial)
    exp_lr_scheduler_custom = lr_scheduler.StepLR(optimizer_custom, step_size=7, gamma=0.1)

    # Train and return trained model
    custom_model = train_model(custom_model, train_loader, valid_loader,
                               optimizer_custom, exp_lr_scheduler_custom, num_epochs=15)

    # Add formatted date time
    torch.save(custom_model.state_dict(), "checkpoints/best_model.pt")

    get_metrics(custom_model, valid_loader)