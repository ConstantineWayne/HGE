from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
gpu_devices = ["0", "1", "2", "3", "4", "5"]
gpu_devices_str = ",".join(gpu_devices)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices_str
import time
import argparse     
from models.models import *
from models.optimization import BertAdam
from utils.eval import get_metrics
from torch.utils.data import DataLoader

from util import get_logger
from dataloaders.cmu_dataloader import AlignedMoseiDataset, UnAlignedMoseiDataset
from models.HingeLoss import HingeLoss
global logger
def get_args(description='Multi-modal Multi-label Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', help="whether to run test")
    parser.add_argument("--aligned", action='store_true', help="whether train align of unalign dataset")
    parser.add_argument('--m3ed', action='store_true')
    parser.add_argument("--data_path", default='./data/train_valid_test.pt', type=str, help='cmu_mosei data_path')
    parser.add_argument("--output_dir", default='./model_saved/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--unaligned_data_path', type=str, default='/data/mosei_senti_data_noalign.pkl',
                        help='load unaligned dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--text_dim', type=int, default=300, help='text_feature_dimension')
    parser.add_argument('--video_dim', type=int, default=35, help='video feature dimension')
    parser.add_argument('--audio_dim', type=int, default=74, help='audio_feature_dimension')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--id', type=int, default=0, help='random seed')
    parser.add_argument("--text_model", default="configs", type=str, required=False, help="text module")
    parser.add_argument("--visual_model", default="configs", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="configs", type=str, required=False, help="Audio module")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--text_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=4, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=4, help="Layer No. of audio")
    parser.add_argument("--num_classes", default=6, type=int, required=False)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--proj_size", type=int, default=64)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--binary_threshold', type=float, default=0.15)
    parser.add_argument('--neg_threshold', type=float, default=0.5)
    parser.add_argument('--pos_threshold', type=float, default=0.5)
    parser.add_argument('--unaligned_mask_same_length', action='store_true')
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--regularization', type=float, default=0.01)
    parser.add_argument('--window_future', type=int, default=4)
    parser.add_argument('--window_past', type=int, default=4)
    parser.add_argument('--vmin', type=float, default=0.03)
    parser.add_argument('--vmax', type=float, default=3.0)
    parser.add_argument('--recon_co', type=float, default=0.1)
    parser.add_argument('--task_co', type=float, default=3.0)
    parser.add_argument('--score_co', type=float, default=1.5)
    parser.add_argument('--time_co', type=float, default=3.0)
    parser.add_argument('--label_co', type=float, default=0.5)
    parser.add_argument('--common_co', type=float, default=0.5)
    parser.add_argument('--cr_co', type=float, default=0.0)  # aligned的设置的0.1
    parser.add_argument('--sim_co', type=float, default=0.1)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--emotion_co', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.03)
    parser.add_argument('--ort_co', type=float, default=1)
    parser.add_argument('--h_co', type=float, default=0.1)  # aligned设置为0.1
    parser.add_argument('--llr', type=float, default=5e-5)
    parser.add_argument('--emo_sim_co', type=float, default=0.1)

    args = parser.parse_args()
    # Check paramenters
    if args.gradient_accumulation_steps < 1: 
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    if args.aligned:
        output_path = os.path.join(args.output_dir, 'aligned')
    else:
        output_path = os.path.join(args.output_dir, 'unaligned')
        if args.unaligned_mask_same_length:
            output_path = os.path.join(output_path, 'same_length')
        else:
            output_path = os.path.join(output_path, 'not_same_length')
    output_path = os.path.join(output_path, 'hge_sd{}_t{}_id{}'.format(args.seed, args.binary_threshold, args.gpu_id))
    args.output_dir = output_path
    return args


def set_seed_logger(args): 
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    torch.cuda.set_device(args.local_rank) 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    return args

def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu
    if args.batch_size % args.n_gpu != 0: 
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
    return device, n_gpu

def prep_optimizer(args, model, num_train_optimization_steps):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    return optimizer, scheduler, model

def prep_dataloader(args):
    if not args.m3ed and args.aligned:
        Dataset = AlignedMoseiDataset
        data_path = args.data_path
    elif not args.m3ed and not args.aligned:
        Dataset = UnAlignedMoseiDataset
        data_path = args.data_path
    elif args.m3ed:
        Dataset = M3ED_Dataset
        data_path = args.m3ed_data_path
    train_dataset = Dataset(
        data_path,
        'train',
        args
    )
    val_dataset = Dataset(
        data_path,
        'valid',
        args
    )
    test_dataset = Dataset(
        data_path,
        'test',
         args
    )
    label_input, label_mask = train_dataset._get_label_input()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        # drop_last=True
    )
    return train_dataloader, val_dataloader, test_dataloader, label_input, label_mask



def save_model(args, model, epoch):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model{}.bin.".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(args, n_gpu, device, model_file=None):
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        model = HGE.from_pretrained(args.text_model, args.visual_model, args.audio_model,
                                       state_dict=model_state_dict, task_config=args)
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0, label_input=None, label_mask=None):
    global logger
    model.train()
    log_step = args.n_display
    total_loss = 0
    total_pred = []
    total_true_label = []
    total_pred_scores = []
    h_loss = HingeLoss()
    cos_loss = nn.CosineEmbeddingLoss()
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        pairs_text, pairs_mask, video, video_mask, audio, audio_mask, ground_label = batch
        res = model(pairs_text, pairs_mask, video, video_mask, audio, audio_mask,
                                                                label_input, label_mask, groundTruth_labels=ground_label, training=True)

        recon_loss_t = res['recon_loss_t']
        recon_loss_v = res['recon_loss_v']
        recon_loss_a = res['recon_loss_a']
        score_loss = res['score_loss']
        label_loss = res['label_loss']
        pred_label = res['pred_label']
        true_label = res['true_label']
        pred_scores = res['predict_scores']
        cr_score_loss = res['cr_score_loss']
        common_score_loss = res['common_score_loss']
        sim_score_loss = res['sim_score_loss']
        time_logits = res['time_logits']
        score_emotion = res['score_emotion']
        loss_ort = res['loss_ort']
        c_sim_t = res['c_sim_t']
        c_sim_v = res['c_sim_v']
        c_sim_a = res['c_sim_a']

        emotion_sim = res['emotion_score_loss']

        ids, feats = [], []
        for i in range(c_sim_t.size(0)):
            feats.append(c_sim_t[i].view(1, -1))
            feats.append(c_sim_v[i].view(1, -1))
            feats.append(c_sim_a[i].view(1, -1))
            ids.append(ground_label[i].view(1, -1))
            ids.append(ground_label[i].view(1, -1))
            ids.append(ground_label[i].view(1, -1))
        feats = torch.cat(feats, dim=0)
        ids = torch.cat(ids, dim=0)

        hinge_loss = h_loss(ids,feats)

        recon_loss = (recon_loss_t + recon_loss_a + recon_loss_v)
        task_loss = torch.nn.BCEWithLogitsLoss()(pred_scores,ground_label.unsqueeze(-2).repeat(1,4,1))
        time_loss = torch.nn.BCEWithLogitsLoss()(time_logits,ground_label)

        model_loss = args.recon_co * recon_loss + args.score_co * score_loss + args.task_co * task_loss + args.label_co * label_loss \
                     + args.cr_co * cr_score_loss + args.common_co * common_score_loss + args.sim_co * sim_score_loss + args.time_co *time_loss + args.emotion_co * score_emotion \
                    + loss_ort * args.ort_co +  hinge_loss * args.h_co + args.emo_sim_co * emotion_sim
        if n_gpu > 1:
            model_loss = model_loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            model_loss = model_loss / args.gradient_accumulation_steps
        model_loss.backward()
        total_loss += float(model_loss)
        total_pred.append(pred_label)
        total_true_label.append(true_label)
        total_pred_scores.append(pred_scores)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %f, task_loss: %f, score_loss:%f, label_loss:%f, loss_ort:%f, emotion_score_loss:%f, time_loss:%f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(model_loss),
                            float(task_loss),
                            float(score_loss),
                            float(label_loss),
                            float(loss_ort),
                            float(score_emotion),
                            float(time_loss))
    total_loss = total_loss / len(train_dataloader)
    total_pred = torch.cat(total_pred, 0)
    total_true_label = torch.cat(total_true_label, 0)
    total_pred_scores = torch.cat(total_pred_scores, 0)
    return total_loss, total_pred, total_true_label, total_pred_scores


def eval_epoch(args, model, val_dataloader, device, n_gpu, label_input, label_mask):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    with torch.no_grad():
        total_pred = []
        total_true_label = []
        total_pred_scores = []
        for _, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
            return_list = model(text, text_mask, video, video_mask, audio, audio_mask,
                                                        label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False)
            true_label = return_list['true_label']
            batch_pred = return_list['pred_label']
            pred_scores = return_list['predict_scores']
            total_true_label.append(true_label)
            total_pred.append(batch_pred)
            total_pred_scores.append(pred_scores)
        total_pred = torch.cat(total_pred, 0)
        total_true_label = torch.cat(total_true_label, 0)
        total_pred_scores = torch.cat(total_pred_scores, 0)
        return total_pred, total_true_label, total_pred_scores
           
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    model = HGE.from_pretrained(args.text_model, args.visual_model, args.audio_model,
                                       task_config=args)
    model = model.to(device)

    if args.do_train:
        train_dataloader, val_dataloader, test_dataloader, label_input, label_mask = prep_dataloader(args)
        label_input = label_input.to(device)
        label_mask = label_mask.to(device)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.000
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            total_loss, total_pred, total_label, total_pred_scores = train_epoch(epoch, args, model, train_dataloader,
                                                                                 device, n_gpu, optimizer,
                                                                                 scheduler, global_step,
                                                                                 local_rank=args.local_rank,
                                                                                 label_input=label_input,
                                                                                 label_mask=label_mask)
            if args.local_rank == 0:
                logger.info("Epoch %d/%d Finished, Train Loss: %f.",
                            epoch + 1, args.epochs, total_loss)

            total_micro_f1, total_micro_precision, total_micro_recall, total_acc, total_macro_f1, total_hl = get_metrics(
                total_pred, total_label)
            if args.local_rank == 0:
                logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f,\tmacro %f, \tEMR: %f", total_micro_f1,
                            total_micro_precision, total_micro_recall, total_acc, total_macro_f1, total_hl)
            if args.local_rank == 0:
                logger.info("***** Running valing *****")
                val_pred, val_label, val_pred_scores = eval_epoch(args, model, val_dataloader, device, n_gpu,
                                                                  label_input, label_mask)
                val_micro_f1, val_micro_precision, val_micro_recall, val_acc, val_macro_f1, val_hl = get_metrics(
                    val_pred, val_label)
                logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f, \tmacro %f, \tEMR %f",
                            val_micro_f1, val_micro_precision, val_micro_recall, val_acc, val_macro_f1, val_hl)
            if args.local_rank == 0:
                test_pred, test_label, test_pred_scores = eval_epoch(args, model, test_dataloader,
                                                                     device, n_gpu, label_input, label_mask)
                test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_macro_f1, test_hl = get_metrics(
                    test_pred,
                    test_label)
                comp_score = test_micro_f1
                output_model_file = save_model(args, model, epoch)
                if best_score <= comp_score:
                    best_score = comp_score
                    best_output_model_file = output_model_file
        if args.local_rank == 0:
            logger.info('***** Running testing *****')
            best_model = load_model(args, n_gpu, device, model_file=best_output_model_file)
            test_pred, test_label, test_pred_scores = eval_epoch(args, best_model, test_dataloader,
                                                                 device, n_gpu, label_input, label_mask)
            test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_macro_f1, test_hl = get_metrics(
                test_pred, test_label)
            logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f,\tmacro %f, \tEMR %f",
                        test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_macro_f1, test_hl)

      
if __name__ == "__main__":
    main()



