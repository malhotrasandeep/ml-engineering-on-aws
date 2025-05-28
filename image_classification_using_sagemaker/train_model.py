import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

import argparse
import logging
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(model, test_loader, criterion, device, loop_type="Test", epoch=-1):
    logger.info(f"Entering test for {loop_type} for epoch {epoch}")
    hook = smd.get_hook(create_if_not_exists=True)
    logger.info(f"In test, hook is of type: {type(hook)}")
 
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # get the output
            outputs = model(inputs)
            # get the prediction
            _, preds = torch.max(outputs, 1)

            # get the loss
            loss = criterion(outputs, labels)

            # calculate the running loss and correct predictions
            # since loss.item() gives average loss for the batch, we multiply by no.
            # of data points in the batch
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    # calculate the total loss and accuracy of the model
    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)

    logger.info(f"Epoch: {epoch} --> {loop_type} Loss: {total_loss}")
    logger.info(f"Epoch: {epoch} --> {loop_type} Accuracy: {100*total_acc}")

    logger.info("Exiting Test")


def train(model, train_loader, val_loader, criterion, optimizer, device):
    logger.info("Entering train")
    hook = smd.get_hook(create_if_not_exists=True)
    logger.info(f"In train, hook is of type: {type(hook)}")

    epochs = 15

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}")
        # set the model for training
        model.train()
        hook.set_mode(smd.modes.TRAIN)
        running_loss = 0.0
        running_corrects = 0
        running_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # get the output of the model for the input
            outputs = model(inputs)

            # calculater the loss
            loss = criterion(outputs, labels)

            # calculate the gradients
            loss.backward()

            # update the gradients
            optimizer.step()

            # get the predictions
            _, preds = torch.max(outputs, 1)
            # calculate the running loss
            # since loss.item() will return the avg. loss for the batch, we
            # will multiply it with the size of the batch to get total loss
            running_loss += loss.item() * inputs.size(0)

            # how many have been predicted correctly
            running_corrects += torch.sum(preds == labels.data).item()

            # size of data
            running_samples += len(inputs)
            if running_samples % 500 == 0:
                accuracy = running_corrects / running_samples
                logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                        running_samples,
                        len(train_loader.dataset),
                        100.0 * (running_samples / len(train_loader.dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                    )
                )

        epoch_loss = running_loss / running_samples
        epoch_acc = running_corrects / running_samples

        logger.info(f"Epoch: {epoch}, Training Accuracy: {100*epoch_acc}, Training Loss: {epoch_loss}")
        logger.info("will validate now")
        test(model, val_loader, criterion, device, "Validation", epoch)
    return model


def net():
    logger.info("Entering net")

    # load the lightweight resnet model with 18 layers
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # set requires_grade for all the parametes to False, as we will be changing
    # only the last layer
    # this is more of feature extraction than fine-tuning. we can do fine-tuning
    # by setting the requires_grad = True for one or more layers of the model
    for param in model.parameters():
        param.requires_grad = False

    # since we are changing the last layer, get the number of input features to that layer
    num_features = model.fc.in_features

    # replate the last layer with num of features equal to the number of dog breeds
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    logger.info("Exiting net")
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # resnet18 needs normalized images of 224x224 size, 
    # we will use the same mean as that used for ImageData training as dog images 
    # are similar
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get the dataset 
    ds = datasets.ImageFolder(root=data, transform=transform)

    # create data set loader
    ds_loader = DataLoader(ds, batch_size=batch_size,shuffle=True)


    return ds_loader


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device is {device}")
    logger.info(f"Hyperparameter values -> Learning Rate = {args.learning_rate}, Batch Size = {args.batch_size}")
    '''
    Initialize a model by calling the net function
    '''
    model = net()

    # create hook
    hook = smd.Hook.create_from_json_file()
    logger.info(f"Created hook. Type is {type(hook)}")
    hook.register_module(model)

    '''
    Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    hook.register_loss(loss_criterion)

    '''
    Call the train function to start training your model
    '''
    train_loader = create_data_loaders(args.train, args.batch_size)
    val_loader = create_data_loaders(args.validation, args.batch_size)
    model = train(model, train_loader, val_loader, loss_criterion, optimizer, device)

    '''
    Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.test, args.batch_size)
    test(model, test_loader, loss_criterion, device)

    '''
    Save the trained model
    '''
    logger.info("Saving the model")
    model_dir = "/opt/ml/model"
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=.01,
        metavar="N",
        help="input learning rate for training (default: .01)",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument(
        "--validation",
        type=str,
        default=os.getenv("SM_CHANNEL_VALIDATION"))
    parser.add_argument(
        "--test",
        type=str,
        default=os.getenv("SM_CHANNEL_TEST"))

    args = parser.parse_args()
    logger.info(f"Starting with hyperparametes --> Batch Size: {args.batch_size} and Learning Rate: {args.learning_rate}")

    main(args)
