from torch import nn
import torch
from torch._C import device
import torch.functional as F
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, n_classes)
        self.input_size = input_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train()

    def forward(self, x):
        x = x.view(-1, self.input_size).to(device='cuda')
        out = self.linear(x)
        return out
    
    def training_step(self, img, organ):
        organ = organ.to(device='cuda')
        out = self(img)                  # Generate predictions
        loss = self.criterion(out, organ) # Calculate loss
        return loss
    
    
    def validation_step(self, img, organ):
        organ = organ.to(device='cuda')
        out = self(img)                    # Generate predictions
        loss = self.criterion(out, organ)   # Calculate loss
        acc = accuracy(out, organ)           # Calculate accuracy
        return {'val_loss': loss.item(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = np.mean(batch_losses)   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = np.mean(batch_accs)      # Combine accuracies
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))