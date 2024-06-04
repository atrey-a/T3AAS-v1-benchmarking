import torch
import numpy as np
from utils.pprint import pprint_verification_start, pprint_verification_end


def verify_model(model, model_device, test_dataloader, skilled_forgeries_dataloader=None, mode='random'):

    model.eval()

    pprint_verification_start(mode)

    genuine_scores = []
    imposter_scores = []

    with torch.inference_mode():
        if mode == 'random':
            for b, [data, label] in enumerate(test_dataloader):
                out = model(data.to(model_device)).detach().cpu()
                out = torch.softmax(out,dim=1)
                for datapoint_id in range(out.shape[0]):
                    for class_id in range(out.shape[1]):
                        if class_id == label[datapoint_id]:
                            genuine_scores.append(out[datapoint_id][class_id])
                        else:
                            imposter_scores.append(out[datapoint_id][class_id])
        elif mode == 'skilled':
            for b, [data, label] in enumerate(test_dataloader):
                out = model(data.to(model_device)).detach().cpu()
                out = torch.softmax(out,dim=1)
                for datapoint_id in range(out.shape[0]):
                    for class_id in range(out.shape[1]):
                        if class_id == label[datapoint_id]:
                            genuine_scores.append(out[datapoint_id][class_id])
            for b, [data, label] in enumerate(skilled_forgeries_dataloader):
                out = model(data.to(model_device)).detach().cpu()
                out = torch.softmax(out,dim=1)
                for datapoint_id in range(out.shape[0]):
                    for class_id in range(out.shape[1]):
                        if class_id == label[datapoint_id]:
                            imposter_scores.append(out[datapoint_id][class_id])

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    X = np.linspace(min(np.concatenate((genuine_scores, imposter_scores))), max(np.concatenate((genuine_scores, imposter_scores))), 1000)
    Nv, _ = np.histogram(genuine_scores, bins=X)
    Nnv, _ = np.histogram(imposter_scores, bins=X)

    frr = np.cumsum(Nv) / len(genuine_scores)
    far = np.cumsum(Nnv[::-1])[::-1] / len(imposter_scores)

    ind = np.argmin(np.abs(far - frr))
    eer = (far[ind] + frr[ind]) / 2

    pprint_verification_end(eer)

    return far, frr, genuine_scores, imposter_scores, eer
