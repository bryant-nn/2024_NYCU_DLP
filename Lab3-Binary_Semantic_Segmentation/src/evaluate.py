from torch.utils import data
from utils import dice_score
import torch
# from utils import load_dataset

def evaluate(net, validation_loader, device, writer, epoch):
    # implement the evaluation function here

    with torch.no_grad():
        net.eval()
        running_loss = 0.0
        running_score = 0

        for i, datas in enumerate(validation_loader):
            inputs, masks = datas['image'], datas['mask']
            inputs, masks = inputs.to(device), masks.to(device)

            outputs = net(inputs)
            score = dice_score(outputs, masks)
            loss = 1 - score

            running_loss += loss.sum().item()
            running_score += score.sum().item()

        epoch_loss = running_loss / len(validation_loader.dataset)
        avg_score = running_score / len(validation_loader.dataset)


    return avg_score, epoch_loss