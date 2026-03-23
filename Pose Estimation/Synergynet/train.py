from model import *
from utils import *
import random
import time

class AverageMeter(object):
    """Tracks and updates running average of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the average with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_dataset, model, optimizer, num_steps, batch_size):
    """Network training, loss updates, and backward calculation"""

    # AverageMeter for statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_name = list(model.get_losses())
    losses_name.append('loss_total')
    losses_meter = [AverageMeter() for i in range(len(losses_name))]

    model.to('mps')
    model.train()

    end = time.time()
    for i in range(num_steps):

        # Randomly sample indices for the batch
        indices = random.sample(range(len(train_dataset)), batch_size)
        input_batch = [train_dataset[idx][0] for idx in indices]
        target_batch = [train_dataset[idx][1] for idx in indices]
        input = torch.stack(input_batch).to('mps')
        target = torch.stack(target_batch)
        target = target[:,:62]
        target.requires_grad = False  
        target = target.float().to('mps')
        
        losses = model(input, target)

        data_time.update(time.time() - end)

        loss_total = 0
        for j, name in enumerate(losses):
            mean_loss = losses[name].mean()
            losses_meter[j].update(mean_loss.item(), input.size(0))
            loss_total += mean_loss

        losses_meter[-1].update(loss_total.item(), input.size(0))
        
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 200 == 0:
            msg = 'Step: [{}/{}]\t'.format(i, num_steps) + \
                  'Time: {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)
            for k in range(len(losses_meter)):
                msg += '{}: {:.4f} ({:.4f})\t'.format(losses_name[k], losses_meter[k].val, losses_meter[k].avg)
            print(msg)

    return model

def main(root,param_fp,filelists,steps=10000,batch_size=32):
    """ Main funtion for the training process"""

    model = SynergyNet()
    optimizer = torch.optim.Adam(model.parameters())
    # normalize = Normalize(mean=127.5, std=128)

    train_dataset = DDFADataset(
        root=root,
        filelists=filelists,
        param_fp=param_fp,
        transform=Compose_GT([ToTensor()]) #normalize
    )
        
    trained_model = train(train_dataset, model, optimizer, steps, batch_size)
    return trained_model