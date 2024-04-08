import argparse
from oxford_pet import load_dataset
from models.unet import Unet
from torch.utils.tensorboard import SummaryWriter
from oxford_pet import load_dataset
import torch
import torch.optim as optim
from torch.utils import data
from utils import dice_score
from evaluate import evaluate
import torch.nn as nn

# global variable
DEVICE = torch.device("cuda" if True and torch.cuda.is_available() else "cpu")
MAX_ACCURACY = 0
LOG_DIR = 'log/Unet'

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        score = dice_score(pred, target)
        
        return 1 - score


def train(args, model, writer):
    # implement the training function here
    global MAX_ACCURACY
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    training_data = load_dataset(args.data_path, 'train')
    training_loader = data.DataLoader(dataset=training_data, batch_size=args.batch_size, shuffle=True)
    criterion = DiceLoss()

    for epoch in range(1, args.epochs + 1):
        # model.train()
        running_loss = 0.0
        running_score = 0

        for i, datas in enumerate(training_loader):
            inputs, masks = datas['image'], datas['mask']
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            masks = masks.to(DEVICE)

            score = dice_score(outputs, masks)
            loss = criterion(outputs, masks)
            # print(loss.shape)
            loss = loss.mean()
            # print(type(loss))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * args.batch_size
            running_score += score.sum().item()

        epoch_loss = running_loss / len(training_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        avg_score = running_score / len(training_loader.dataset)
        writer.add_scalar('Dice Score/train', avg_score, epoch)

        print(f'Train Epoch {epoch}, Loss: {epoch_loss}, Dice Score: {avg_score}')


        # validation
        valid_data = load_dataset(args.data_path, 'valid')
        valid_loader = data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
        
        avg_score, epoch_loss = evaluate(model, valid_loader, DEVICE, writer, epoch)

        print(f'Valid Epoch {epoch}, Loss: {epoch_loss}, Dice Score: {avg_score}')
        print('=====================================')

        writer.add_scalar('Loss/valid', epoch_loss, epoch)
        writer.add_scalar('Dice Score/valid', avg_score, epoch)

        # save model
        if avg_score > MAX_ACCURACY and avg_score > 0.8:
            MAX_ACCURACY = avg_score
            torch.save(model.state_dict(), f'{LOG_DIR}/best_model_{avg_score}.pth')

    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default='../dataset/oxford-iiit-pet', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=12, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    # MAX_ACCURACY = 0
    args = get_args()
    writer = SummaryWriter(LOG_DIR)
    model = Unet(3, 1).to(DEVICE)
    train(args, model, writer)