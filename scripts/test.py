import time
import torch
from torch.nn.functional import one_hot
from utils.pprint import pprint_epoch_header, pprint_epoch_footer, pprint_verbose_table


def test_model(e, E, model, dataloader, num_classes, loss_function, model_device, loop_device, verbose=False, mode='val'):
    loss, acc = 0., 0.

    pprint_epoch_header(e,E,verbose,mode)

    model.eval()

    B = len(dataloader)

    with torch.inference_mode():
        start_time = time.time()
        for b, [data, label] in enumerate(dataloader):
            out = model(data.to(model_device)).to(loop_device)
            target = one_hot(label.to(torch.int64), num_classes).to(dtype=torch.float32, device=loop_device)
            accu = torch.sum(torch.argmax(out,-1)==torch.argmax(target,1))
            acc += accu.item()
            los = loss_function(out, target)
            loss += los.item()
            if verbose:
                pprint_verbose_table(e,E,b+1,B,label,torch.argmax(out,-1),los)
        time_taken = time.time() - start_time

    loss /= float(len(dataloader.dataset))
    acc /= float(len(dataloader.dataset))
    time_taken /= float(len(dataloader.dataset))

    pprint_epoch_footer(e,E,acc,loss,verbose,mode)

    return loss, acc, time_taken*1000000
