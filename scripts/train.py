import torch
from torch.nn.functional import one_hot
from utils.pprint import pprint_epoch_header, pprint_epoch_footer, pprint_verbose_table


def train_epoch(e, E, model, dataloader, num_classes, loss_function, optimizer, model_device, loop_device, verbose=False):
    loss, acc = 0., 0.

    pprint_epoch_header(e,E,verbose)

    model.train()

    B = len(dataloader)

    for b, [data, label] in enumerate(dataloader):
        out = model(data.to(model_device)).to(loop_device)
        target = one_hot(label.to(torch.int64), num_classes).to(dtype=torch.float32, device=loop_device)
        accu = torch.sum(torch.argmax(out,-1)==torch.argmax(target,1))
        acc += accu.item()
        optimizer.zero_grad()
        los = loss_function(out, target)
        loss += los.item()
        los.backward()
        optimizer.step()
        if verbose:
            pprint_verbose_table(e,E,b+1,B,label,torch.argmax(out,-1),los)
    loss /= float(len(dataloader.dataset))
    acc /= float(len(dataloader.dataset))

    pprint_epoch_footer(e,E,acc,loss,verbose)

    return model, loss, acc
