import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from datetime import datetime
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import datasets.lmas, datasets.svc, datasets.t3aas
import models.simple_cnn, models.simple_rnn, models.two_level_rnn, models.slitcnn, models.slitcnn_lstm, models.vgg16
from scripts.train import train_epoch
from scripts.test import test_model
from scripts.verify import verify_model
from utils.collate import collate_by_center_padding, collate_by_end_padding, collate_by_stripping_pen_tail
from utils.logger import create_csv, add_to_csv
from utils.writer import create_tb_log, log_scalar
from utils.pprint import pprint_results, pprint_line
from utils.pbar import progress_bar

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--logs', type=str, default='./outputs', help='Path to the directory that stores logs, results and checkpoints for all experiments (defaults to \'./outputs\')')

parser.add_argument('--model', type=str, required=True, choices=['simple_gru', 'simple_lstm', '2level_gru', '2level_lstm', 'vgg_16', 'cnn_1d', 'slitcnn', 'twostream_slitcnn', 'twostream_slitcnn_lstm_parallel', 'twostream_slitcnn_lstm_sequential', 'slitcnn_lstm'], help='Name of the model')
parser.add_argument('--dataset', type=str, required=True, choices=['t3aas', 'svc', 'lmas'], help='Name of the dataset')
parser.add_argument('--mode', type=str, required=True, choices=['raw_form', 'images', 'padded_form', 'features'], default='raw_form', help='Name of the dataset')
parser.add_argument('--pen_tip', action="store_true", help='Use only the pen-tip from the t3aas dataset')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the dataloader')

parser.add_argument('--test_each_epoch', action="store_true", help='Test after every epoch (As opposed to testing the best epoch at the end)')
parser.add_argument('--verify_random_forgery', action="store_true", help='Run verification for random forgery after training')
parser.add_argument('--verify_skilled_forgery', action="store_true", help='Run verification for skilled forgery after training')

parser.add_argument('--verbose_print', action="store_true", help='Pretty print every datapoint\'s predictions')
parser.add_argument('--print_to_file', action="store_true", help='Print the outputs to a file (\'output.txt\' in the --logs directory) instead of stdout')

parser.add_argument('--dataloader_device', type=str, choices=['cpu', 'cuda', *[f'cuda:{i}' for i in range(8)]], default='cpu', help='Device to store tensors in the dataloader')
parser.add_argument('--model_device', type=str, choices=['cpu', 'cuda', *[f'cuda:{i}' for i in range(8)]], default='cuda:0', help='Device to store the model in')
parser.add_argument('--training_device', type=str, choices=['cpu', 'cuda', *[f'cuda:{i}' for i in range(8)]], default='cuda:1', help='Device to store tensors in the train loop')

args = parser.parse_args()

if __name__ == '__main__':
    exp_name = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{args.model}_on_{args.mode}_of_{args.dataset}_at_{os.path.basename(args.path)}{'_pen_tip' if args.pen_tip else ''}"
    out_folder = os.path.join(args.logs, exp_name)
    os.makedirs(out_folder)

    with open(os.path.join(out_folder,'config.json'),'w') as f:
        json.dump(args.__dict__, f, indent=4, sort_keys=False)

    if args.print_to_file:
        sys.stdout = open(os.path.join(out_folder, "output.txt"), "w")

    if args.dataset == 't3aas':
        num_classes = 45
        if args.mode == 'raw_form':
            if not args.pen_tip:
                train_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
                val_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
                test_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
                if args.verify_skilled_forgery:
                    skilled_forgeries_dataloader = DataLoader(
                        datasets.t3aas.T3AAS(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device)),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                    )
                if args.model == 'slitcnn':
                    model = models.slitcnn.OneStreamSlitCNN(512,6,num_classes)
                elif args.model == 'twostream_slitcnn':
                    model = models.slitcnn.TwoStreamSlitCNN(512,6,num_classes)
                elif args.model == 'slitcnn_lstm':
                    model = models.slitcnn_lstm.SlitCNN_with_LSTM(512,6,num_classes)
                elif args.model == 'twostream_slitcnn_lstm_sequential':
                    model = models.slitcnn_lstm.TwoStream_SlitCNN_with_LSTM_Sequential(512,6,num_classes)
                elif args.model == 'twostream_slitcnn_lstm_parallel':
                    model = models.slitcnn_lstm.TwoStream_SlitCNN_with_LSTM_Parallel(512,6,num_classes)
                else:
                    raise Exception(f"{args.model} is not compatible with {args.mode} of {args.dataset} with pen_tip {args.pen_tip}")
            else:
                train_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_stripping_pen_tail(0,2)
                )
                val_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_stripping_pen_tail(0,2)
                )
                test_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_stripping_pen_tail(0,2)
                )
                if args.verify_skilled_forgery:
                    skilled_forgeries_dataloader = DataLoader(
                        datasets.t3aas.T3AAS(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device)),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        collate_fn=collate_by_stripping_pen_tail(0,2)
                    )
                if args.model == 'slitcnn':
                    model = models.slitcnn.OneStreamSlitCNN(512,3,num_classes)
                elif args.model == 'slitcnn_lstm':
                    model = models.slitcnn_lstm.SlitCNN_with_LSTM(512,3,num_classes)
                else:
                    raise Exception(f"{args.model} is not compatible with {args.mode} of {args.dataset} with pen_tip {args.pen_tip}")
        elif args.mode == 'images':
            train_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Images(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Images(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Images(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            if args.verify_skilled_forgery:
                skilled_forgeries_dataloader = DataLoader(
                    datasets.t3aas.T3AAS_Images(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
            if args.model == 'vgg_16':
                model = models.vgg16.VGG16(num_classes)
            else:
                raise Exception(f"{args.model} is not compatible with {args.mode} of {args.dataset}")
        elif args.mode == 'padded_form':
            if args.model == 'simple_lstm' or args.model == 'simple_gru':
                train_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_end_padding(330)
                )
                val_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_end_padding(330)
                )
                test_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_end_padding(330)
                )
                if args.verify_skilled_forgery:
                    skilled_forgeries_dataloader = DataLoader(
                        datasets.t3aas.T3AAS(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device)),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        collate_fn=collate_by_end_padding(330)
                    )
                if args.model == 'simple_lstm':
                    model = models.simple_rnn.SimpleLSTM(330,12,num_classes)
                elif args.model == 'simple_gru':
                    model = models.simple_rnn.SimpleGRU(330,12,num_classes)
            else:
                train_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_center_padding(330)
                )
                val_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_center_padding(330)
                )
                test_dataloader = DataLoader(
                    datasets.t3aas.T3AAS(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device)),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                    collate_fn=collate_by_center_padding(330)
                )
                if args.verify_skilled_forgery:
                    skilled_forgeries_dataloader = DataLoader(
                        datasets.t3aas.T3AAS(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device)),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True,
                        collate_fn=collate_by_center_padding(330)
                    )
                if args.model == 'cnn_1d':
                    model = models.simple_rnn.SimpleLSTM(330,12,num_classes)
                else:
                    raise Exception(f"{args.model} is not compatible with {args.mode} of {args.dataset}")
        elif args.mode == 'features':
            train_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Features(os.path.join(args.path,'Train'), loader_device=torch.device(args.dataloader_device), num_windows=8, window_size=64, num_columns=12),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            val_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Features(os.path.join(args.path,'Validation'), loader_device=torch.device(args.dataloader_device), num_windows=8, window_size=64, num_columns=12),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            test_dataloader = DataLoader(
                datasets.t3aas.T3AAS_Features(os.path.join(args.path,'Test'), loader_device=torch.device(args.dataloader_device), num_windows=8, window_size=64, num_columns=12),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
            )
            if args.verify_skilled_forgery:
                skilled_forgeries_dataloader = DataLoader(
                    datasets.t3aas.T3AAS_Features(os.path.join(args.path,'SkilledForgeries'), loader_device=torch.device(args.dataloader_device), num_windows=8, window_size=64, num_columns=12),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
            if args.model == '2level_lstm':
                model = models.two_level_rnn.TwoLevelLSTM(12,8,num_classes,6)
            elif args.model == '2level_gru':
                model = models.two_level_rnn.TwoLevelGRU(12,8,num_classes,6)
            else:
                raise Exception(f"{args.model} is not compatible with {args.mode} of {args.dataset}")

    model.to(torch.device(args.model_device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(),lr=args.lr)

    val_accs, val_losses = [], []
    test_accs, test_losses, test_times = [], [], []

    best_val_epoch, best_val_acc = 0, 0.
    best_test_epoch, best_test_acc = 0, 0.

    create_csv(os.path.join(out_folder,'results.csv'))
    writer = create_tb_log(os.path.join(out_folder,'tensorboard_logs'))

    prog = progress_bar(args.print_to_file, exp_name)

    for epoch in prog(range(args.epochs)):
        model, train_loss, train_acc = train_epoch(epoch+1, args.epochs, model, train_dataloader, num_classes, criterion, optimizer, torch.device(args.model_device), torch.device(args.training_device), args.verbose_print)
        add_to_csv(os.path.join(out_folder,'results.csv'),'train',epoch+1,train_loss,train_acc)
        val_loss, val_acc, _ = test_model(epoch+1, args.epochs, model, val_dataloader, num_classes, criterion, torch.device(args.model_device), torch.device(args.training_device), args.verbose_print, mode='val')
        add_to_csv(os.path.join(out_folder,'results.csv'),'val',epoch+1,val_loss,val_acc)
        log_scalar(writer, 'train/loss', train_loss, epoch+1)
        log_scalar(writer, 'train/acc', train_acc, epoch+1)
        log_scalar(writer, 'val/loss', val_loss, epoch+1)
        log_scalar(writer, 'val/acc', val_acc, epoch+1)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        if args.test_each_epoch:
            test_loss, test_acc, test_time = test_model(epoch+1, args.epochs, model, test_dataloader, num_classes, criterion, torch.device(args.model_device), torch.device(args.training_device), args.verbose_print, mode='test')
            add_to_csv(os.path.join(out_folder,'results.csv'),'test',epoch+1,test_loss,test_acc)
            log_scalar(writer, 'test/loss', test_loss, epoch+1)
            log_scalar(writer, 'test/acc', test_acc, epoch+1)
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            test_times.append(test_time)
            if test_acc > best_test_acc:
                best_test_epoch = epoch+1
                best_test_acc = test_acc
                checkpoint = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'test_acc': test_acc,
                    'test_loss': test_loss,
                }
                torch.save(checkpoint, os.path.join(out_folder,f"best_test_checkpoint.pth"))
        elif val_acc > best_val_acc:
            best_val_epoch = epoch+1
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(out_folder,f"best_val_checkpoint.pth"))

        writer.close()
        pprint_line()

    ptest_acc = 0.
    ptest_time = 0

    if not args.test_each_epoch:
        testing_checkpoint = torch.load(os.path.join(out_folder,f"best_val_checkpoint.pth"))
        model.load_state_dict(testing_checkpoint['model_state_dict'])
        test_loss, test_acc, test_time = test_model(testing_checkpoint['epoch'], args.epochs, model, test_dataloader, num_classes, criterion, torch.device(args.model_device), torch.device(args.training_device), args.verbose_print, mode='test')
        add_to_csv(os.path.join(out_folder,'results.csv'),'test',testing_checkpoint['epoch'],test_loss,test_acc)
        ptest_acc = test_acc
        ptest_time = test_time
        pprint_line()
    else:
        ptest_acc = best_test_acc
        ptest_time = sum(test_times)/len(test_times)

    eer_random = None
    eer_skilled = None

    if args.test_each_epoch:
        verification_checkpoint = torch.load(os.path.join(out_folder,f"best_test_checkpoint.pth"))
    else:
        verification_checkpoint = torch.load(os.path.join(out_folder,f"best_val_checkpoint.pth"))

    model.load_state_dict(verification_checkpoint['model_state_dict'])

    if args.verify_random_forgery:
        add_to_csv(os.path.join(out_folder,'results.csv'),'mode','random/skilled','random_eer','skilled_eer')
        *_, genuine_scores, imposter_scores, eer_random = verify_model(model, torch.device(args.model_device), test_dataloader, mode='random')
        add_to_csv(os.path.join(out_folder,'results.csv'),'verify','random',eer_random,0)
        np.save(os.path.join(out_folder,f"genuine_scores_random_forgery.npy"), genuine_scores)
        np.save(os.path.join(out_folder,f"imposter_scores_random_forgery.npy"), imposter_scores)

    if args.verify_skilled_forgery:
        add_to_csv(os.path.join(out_folder,'results.csv'),'mode','random/skilled','random_eer','skilled_eer')
        *_, genuine_scores, imposter_scores, eer_skilled = verify_model(model, torch.device(args.model_device), test_dataloader, skilled_forgeries_dataloader, mode='skilled')
        add_to_csv(os.path.join(out_folder,'results.csv'),'verify','skilled',0,eer_skilled)
        np.save(os.path.join(out_folder,f"genuine_scores_skilled_forgery.npy"), genuine_scores)
        np.save(os.path.join(out_folder,f"imposter_scores_skilled_forgery.npy"), imposter_scores)

    pprint_line()

    pprint_results(ptest_acc, sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad), ptest_time, eer_random, eer_skilled)

    if args.print_to_file:
        sys.stdout.close()
        sys.stdout = sys.__stdout__
