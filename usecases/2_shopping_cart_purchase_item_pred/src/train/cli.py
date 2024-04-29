import argparse
import numpy as np
import torch
import random
import sys
import os, sys
from sihg4sr.utils import load_file, prepare_candidate_filter, Timer
from sihg4sr.model import SIHG4SR_Handler
from sihg4sr.dataloader import SIHG4SR_DataLoader
import pickle

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [
        int(x.split()[2]) for x in open('tmp', 'r').readlines()
    ]
    # memory_available = memory_available[1:6]
    if len(memory_available) == 0:
        return -1
    return int(np.argmax(memory_available))

def main(args):
    seed_torch(123)

    device = "cpu"
    if device == 'gpu':
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_freer_gpu())
        print(f"CUDA_VISIBLE_DEVICES is {os.environ['CUDA_VISIBLE_DEVICES']}")
    elif device == 'hpu':
        pass

    if os.path.exists(args.feat_dir):
        with open(os.path.join(args.feat_dir, "meta.pkl"), 'rb') as file:
            metadata = pickle.load(file)
    num_items = metadata['num_items']
    num_unique_features = metadata['num_unique_features']
    time_feature = metadata['time_feature']
    feature_list = metadata['feature_list']
    
    candidate_df_path = os.path.join(args.dataset_dir, "candidate_items.csv")
    if os.path.exists(candidate_df_path):
        logit_difference_batch = prepare_candidate_filter(
            candidate_df=load_file(candidate_df_path),
            num_items=num_items,
            batch_size=args.batch_size
        )
    else:
        logit_difference_batch = None

    model_hparams = {
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "dropout": args.feat_drop,
        "reducer": args.reducer,
        "order": args.order,
        "norm": args.norm,
        "extra": True,
        "fusion": args.fusion,
        "srl_ratio": args.srl_ratio,
        "srg_ratio": args.srg_ratio,
    }
    model = SIHG4SR_Handler(
        num_items=num_items,
        num_unique_features=num_unique_features,
        num_categories=74,
        model_hparams=model_hparams,
        resumed_model=args.model,
        save_path=args.save_path,
        topk=args.topk,
        filter=logit_difference_batch,
        finetune=args.finetune,
        device=device,
    )
    
    if args.test:
        with Timer("Test"):
            sessions_df = load_file(os.path.join(args.dataset_dir, "valid_processed.parquet"))
            valid_dataloader = SIHG4SR_DataLoader(
                sessions=sessions_df,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                order=args.order,
                attent_longest_view=args.attent_longest_view,
                enable_sampler=False,
                time_feature=time_feature,
                feature_list=feature_list,
                recent_n_month=-1,
                sort=False
            )
            mrr, hit = model.evaluate(valid_dataloader)
            print('Test:')
            print(f'\tRecall@{args.topk}:\t{hit:.6f}\tMMR@{args.topk}:\t{mrr:.6f}\t')
        return

    if args.predict:
        with Timer("Predict"):
            sessions_df = load_file(os.path.join(args.dataset_dir, "test_processed.parquet"))
            test_dataloader = SIHG4SR_DataLoader(
                sessions=sessions_df,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                order=args.order,
                attent_longest_view=args.attent_longest_view,
                enable_sampler=False,
                time_feature=time_feature,
                feature_list=feature_list,
                recent_n_month=-1,
                sort=False
            )
            preds = model.predict(test_dataloader)
            # TODO: Save preds to local disk
            print(f'Predict completed, file is saved to {args.save_path}')
        return
    
    with Timer("Load Train data"):
        train_sessions_df = load_file(os.path.join(args.dataset_dir, "train_processed.parquet"))
    with Timer("Load Valid data"):
        valid_sessions_df = load_file(os.path.join(args.dataset_dir, "valid_processed.parquet"))
    train_dataloader = SIHG4SR_DataLoader(
        sessions=train_sessions_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=args.order,
        attent_longest_view=args.attent_longest_view,
        enable_sampler=True,
        time_feature=time_feature,
        feature_list=feature_list,
        recent_n_month=args.use_recent_n_month,
        sort=args.sort_train_data
    )
    valid_dataloader = SIHG4SR_DataLoader(
        sessions=valid_sessions_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=args.order,
        attent_longest_view=args.attent_longest_view,
        enable_sampler=False,
        time_feature=time_feature,
        feature_list=feature_list,
        recent_n_month=-1,
        sort=False
    )
    
    print('start training', flush=True)
    mrr, hit = model.train(
        train_loader=train_dataloader, 
        val_loader=valid_dataloader,
        epochs=args.epochs,
        log_interval=args.log_interval)
    
    print('training completed', flush=True)
    print(f'MRR@{args.topk}\tHR@{args.topk}', flush=True)
    print(f'{mrr * 100:.3f}%\t{hit * 100:.3f}%', flush=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--embedding-dim',type=int,default=256,help='the embedding size')
    parser.add_argument('--num-layers',type=int,default=1,help='the number of layers')
    parser.add_argument('--feat-drop',type=float,default=0.1,help='the dropout ratio for features')
    parser.add_argument('--lr', type=float,default=1e-3, help='the learning rate')
    parser.add_argument('--weight-decay',type=float,default=1e-4,help='the parameter for L2 regularization')
    parser.add_argument('--patience',type=int,default=5,help='the number of epochs that the performance does\
        not improves after which the training stops',)
    parser.add_argument('--num-workers',type=int,default=4,help='the number of processes to load the input graphs',)
    parser.add_argument('--valid-split',type=float,default=None,help='the fraction for the validation set',)
    parser.add_argument('--log-interval',type=int,default=100,help='print the loss after this number of iterations',)
    parser.add_argument('--order',type=int,default=1,help='order of msg',)
    parser.add_argument('--reducer',type=str,default='mean',help='method for reducer',)
    parser.add_argument('--norm',type=bool,default=True,help='whether use l2 norm',)
    parser.add_argument('--extra',action='store_true',help='whether use REnorm.',)
    parser.add_argument('--fusion', action='store_true', help='whether use IFR.')
    parser.add_argument('--topk', type=int, default=100, help='topk')
    parser.add_argument('--test-with-random-sample', action='store_true', help='test dataset will be random sampled')
    parser.add_argument('--data-augment', default=1, type=int, help='the scale of data augment')
    parser.add_argument('--use-target-output', action='store_true', help='test')
    parser.add_argument('--enable-features-gnn', action='store_true', help='enable item features to add a knowledge graph to GNN')
    parser.add_argument('--attent-longest-view', action='store_true', help='instead of using last item to add attend, use item with longest view duration instead')
    parser.add_argument('--save-path', default='model_save/', type=str)
    parser.add_argument('--batch-size',type=int,default=512,help='the batch size for training')
    parser.add_argument('--epochs',type=int,default=15,help='the number of training epochs')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--predict', action='store_true', help='predict')
    parser.add_argument('--finetune', action='store_true', help='finetune option would config resume_train, epochs, and use-recent-n-month sort-train-data at once')
    parser.add_argument('--resume-train', action='store_true', help='resume train with existing model')
    parser.add_argument('--model',type=str,default='model_save/0323/model_0_0.168.pth',help='saved model path',)
    parser.add_argument('--dataset-dir',default=f'../../data/',help='the dataset directory')
    parser.add_argument('--feat-dir',default=f'../../data/',help='the features directory')
    parser.add_argument('--use-recent-n-month', default=-1, type=float, help='use recent months of data to train, -1 means not enable, n means recent n months')
    parser.add_argument('--enable-weighted-loss', action='store_true', help='enable weighted loss based on ts')
    parser.add_argument('--sort-train-data', action='store_true', help='sort training data')
    parser.add_argument('--srl_ratio',type=int,default=0.7)
    parser.add_argument('--srg_ratio',type=int,default=0.3)

    args = parser.parse_args()
    
    main(args)