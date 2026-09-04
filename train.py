import numpy as np
import torch, random
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import time
import pandas as pd
from datetime import datetime
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from model.view_gcn import view_GCN, SVCNN
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
# os.environ['CUDA_VISIBLE_DEVICES']='2'
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="view-gcn")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=20)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=0)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="resnet18")
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument("-train_path", type=str, default="data/modelnet40_views/*/train")
parser.add_argument("-val_path", type=str, default="data/modelnet40_views/*/test")
parser.add_argument("--dataset", type=str, choices=['modelnet40', 'scanobjectnn', 'colombia', 'roofn3d'], default="modelnet40", help="Dataset to use")
parser.add_argument("--stage1_epochs", type=int, default=25, help="Number of epochs for stage 1 (SVCNN)")
parser.add_argument("--stage2_epochs", type=int, default=20, help="Number of epochs for stage 2 (view-GCN)")
parser.add_argument("--edge_dim", type=int, choices=[None, 6, 10], default=None, help="Edge feature dimension: None (no edge features), 6 ([vi, vj]), 10 ([vi, vj, vi-vj, |vi-vj|])")
parser.add_argument("--att_lambda", type=float, default=0.3, help="Weight for GAT attention coefficients in saliency scoring (0.0-1.0)")
parser.add_argument("--diff_topk", action='store_true', help="Use differentiable Top-K selection instead of hard selection")
parser.add_argument('--rank_margin', type=float, default=0.2)
parser.add_argument('--att_entropy', type=float, default=0.01, help='entropy penalty weight inside the attention regularizer')
parser.add_argument('--freeze_epochs', type=int, default=5, help='Number of epochs to freeze backbone in Stage 1 (0 means no freezing). Default: 5 when pretraining=True, 0 when pretraining=False')
parser.add_argument('--n_attn_heads', type=str, default='8', help='Number of attention heads for each layer (comma-separated, e.g., 8 or 16,8,2). If single value, used for all levels.')
parser.add_argument('--num_level', type=int, default=3, help='Number of hierarchical levels in Stage 2 view-GCN (1, 2, 3, or more)')
parser.add_argument('--graph_net', type=str, choices=['gat', 'gcn'], default='gat', help='Graph neural network type: gat (Graph Attention) or gcn (Graph Convolution)')
parser.add_argument('--save_exp', type=lambda x: x.lower() != 'false', default=True, help='Save experiment results to exp_result.csv (default: True)')
parser.add_argument('--data_ver', type=str, default=None, help='Data version identifier for experiment tracking')
parser.add_argument('--remark', type=str, default=None, help='Remark/notes for this experiment')
parser.set_defaults(train=False)

def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


if __name__ == '__main__':
    # Start total timing
    total_start_time = time.time()
    
    seed_torch()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    # Set freeze_epochs default: 5 when pretraining=True (fine-tuning), 0 when pretraining=False (train from scratch)
    if not hasattr(args, 'freeze_epochs') or args.freeze_epochs == 5:
        args.freeze_epochs = 5 if pretraining else 0
    # Parse n_attn_heads from comma-separated string to list of integers
    if isinstance(args.n_attn_heads, str):
        args.n_attn_heads = [int(x.strip()) for x in args.n_attn_heads.split(',')]
    # If single value provided, replicate for all levels
    if len(args.n_attn_heads) == 1:
        args.n_attn_heads = args.n_attn_heads * args.num_level
    if len(args.n_attn_heads) != args.num_level:
        raise ValueError(f"--n_attn_heads must have exactly {args.num_level} values (one for each level), got {len(args.n_attn_heads)}: {args.n_attn_heads}")
    log_dir = args.name
    create_folder(args.name)
    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()
    
    print(f"\n{'='*80}")
    print(f"VIEW-GCN TRAINING PIPELINE STARTED")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Experiment name: {args.name}")
    print(f"Batch size: {args.batchSize}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of views: {args.num_views}")
    print(f"CNN backbone: {args.cnn_name}")
    print(f"Stage 1 epochs: {args.stage1_epochs}")
    print(f"Stage 2 epochs: {args.stage2_epochs}")
    print(f"Edge features: {args.edge_dim}D" if args.edge_dim is not None else "Edge features: None")
    print(f"Attention lambda: {args.att_lambda}")
    print(f"Differentiable Top-K: {args.diff_topk}")
    print(f"Backbone freeze epochs (Stage 1): {args.freeze_epochs}")
    print(f"Number of hierarchical levels: {args.num_level}")
    print(f"Graph network type: {args.graph_net.upper()}")
    print(f"Attention heads per layer: {args.n_attn_heads}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
    print(f"{'='*80}\n")
    
    # Set dataset-specific parameters
    if args.dataset == 'modelnet40':
        nclasses = 40
        dataset_name = 'modelnet40'
        # Use provided paths or defaults
        train_path = args.train_path
        val_path = args.val_path
    elif args.dataset == 'scanobjectnn':
        nclasses = 15
        dataset_name = 'scanobjectnn'
        # Auto-set paths for ScanObjectNN rendered views
        print("="*100)
        train_path = "data/scanobjectnn_views/*/train"
        val_path = "data/scanobjectnn_views/*/test"
        print(f"Using ScanObjectNN dataset with {nclasses} classes")
        print(f"Training path: {train_path}")
        print(f"Validation path: {val_path}")
    elif args.dataset == 'colombia':
        nclasses = 6
        dataset_name = 'colombia'
        # Auto-set paths for Colombia dataset
        print("="*100)
        train_path = "data/colombia_views/*/train"
        val_path = "data/colombia_views/*/test"
        print(f"Using Colombia dataset with {nclasses} classes")
        print(f"Training path: {train_path}")
        print(f"Validation path: {val_path}")
    elif args.dataset == 'roofn3d':
        nclasses = 2
        dataset_name = 'roofn3d'
        train_path = "data/roofn3d_views/*/train"
        val_path = "data/roofn3d_views/*/test"
        print(f"Using RoofN3D dataset with {nclasses} classes")
        print(f"Training path: {train_path}")
        print(f"Validation path: {val_path}")
        
    
    # STAGE 1
    stage1_start_time = time.time()
    log_dir = args.name+'_stage_1'
    create_folder(log_dir)
    cnet = SVCNN(f"{args.name}_svcnn", nclasses=nclasses, pretraining=pretraining, cnn_name=args.cnn_name, dataset=dataset_name)
    
    # Freeze backbone before creating optimizer if freeze_epochs > 0
    if args.freeze_epochs > 0 and pretraining:
        if hasattr(cnet, 'freeze_backbone'):
            cnet.freeze_backbone()
            print(f"Backbone frozen before training (will unfreeze after {args.freeze_epochs} epochs)")
    
    # Create optimizer with only trainable parameters
    trainable_params = [p for p in cnet.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check if model was initialized correctly.")
    optimizer = optim.SGD(trainable_params, lr=1e-2, weight_decay=args.weight_decay, momentum=0.9)
    n_models_train = args.num_models*args.num_views
    train_dataset = SingleImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views, dataset=dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    val_dataset = SingleImgDataset(val_path, scale_aug=False, rot_aug=False, test_mode=True, dataset=dataset_name)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=4)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    print(f"The number of classes is {nclasses}")
    trainer = ModelNetTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir, num_views=1, nclasses=nclasses, args=args)
    stage1_time, stage1_best_acc, stage1_best_epoch = trainer.train(args.stage1_epochs)

    # Load the best SVCNN model for stage 2
    print("="*100)
    print("LOADING BEST SVCNN MODEL FOR STAGE 2")
    print("="*100)
    if not cnet.load_best_model(log_dir):
        print("Warning: Best SVCNN model not found, using final epoch model")

    # Aggressively clear CUDA cache and delete stage 1 objects between stages
    if torch.cuda.is_available():
        # Move model to CPU and delete trainer to free memory
        cnet.cpu()
        del trainer
        del train_loader
        del val_loader
        del train_dataset
        del val_dataset
        del optimizer
        # Clear all CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("CUDA memory cleared between stages")

    # # # STAGE 2
    stage2_start_time = time.time()
    print("="*100)
    print("STAGE 2")
    print("="*100)
    log_dir = args.name+'_stage_2'
    create_folder(log_dir)
    # Check if view_GCN accepts n_attn_heads parameter (for backward compatibility)
    import inspect
    sig = inspect.signature(view_GCN.__init__)
    view_gcn_kwargs = {
        'nclasses': nclasses,
        'cnn_name': args.cnn_name,
        'num_views': args.num_views,
        'dataset': dataset_name,
        'edge_dim': args.edge_dim,
        'att_lambda': args.att_lambda,
        'diff_topk': args.diff_topk,
        'num_levels': args.num_level,
        'graph_net': args.graph_net
    }
    # Only add n_attn_heads if the parameter exists in the signature
    if 'n_attn_heads' in sig.parameters:
        view_gcn_kwargs['n_attn_heads'] = args.n_attn_heads
    cnet_2 = view_GCN(f"{args.name}_viewgcn", cnet, **view_gcn_kwargs)
    optimizer = optim.SGD(cnet_2.parameters(), lr=args.lr, weight_decay=args.weight_decay,momentum=0.9)
    train_dataset = MultiviewImgDataset(train_path, scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=args.num_views,test_mode=True, dataset=dataset_name)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer
    val_dataset = MultiviewImgDataset(val_path, scale_aug=False, rot_aug=False, num_views=args.num_views,test_mode=True, dataset=dataset_name)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'view_gcn', log_dir, num_views=args.num_views, nclasses=nclasses, args=args)
    #use trained_view_gcn
    #cnet_2.load_state_dict(torch.load('trained_view_gcn.pth'))
    #trainer.update_validation_accuracy(1)
    stage2_time, stage2_best_acc, stage2_best_epoch = trainer.train(args.stage2_epochs)
    
    # Final evaluation with best model
    print("\n" + "="*100)
    print("FINAL EVALUATION WITH BEST MODEL")
    print("="*100)
    print(f"Stage 1 Best Accuracy: {stage1_best_acc:.4f} (Epoch {stage1_best_epoch})")
    print(f"Stage 2 Best Accuracy: {stage2_best_acc:.4f} (Epoch {stage2_best_epoch})")
    
    # Load best view-GCN model for final evaluation
    if not cnet_2.load_best_model(log_dir):
        print("Warning: Best view-GCN model not found, using final epoch model")
    
    # Run final evaluation using the existing trainer
    print("\n" + "="*60)
    print("RUNNING FINAL EVALUATION")
    print("="*60)
    
    # Set model to evaluation mode
    cnet_2.eval()
    
    # Run evaluation using the existing trainer's method
    with torch.no_grad():
        (final_loss, final_val_overall_acc, final_val_mean_class_acc, final_macro_f1, final_f1_scores_per_class,
         final_macro_precision, final_macro_recall, final_micro_precision, final_micro_recall, final_micro_f1,
         final_weighted_precision, final_weighted_recall, final_weighted_f1, final_precision_per_class, final_recall_per_class) = trainer.update_validation_accuracy(0)
    
    # Calculate total time and print final summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\n{'='*80}")
    print(f"VIEW-GCN TRAINING PIPELINE COMPLETED")
    print(f"{'='*80}")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Stage 1: {args.stage1_epochs} epochs, {stage1_time:.2f} seconds ({stage1_time/60:.2f} minutes)")
    print(f"  - Best Accuracy: {stage1_best_acc:.4f} (Epoch {stage1_best_epoch})")
    print(f"Stage 2: {args.stage2_epochs} epochs, {stage2_time:.2f} seconds ({stage2_time/60:.2f} minutes)")
    print(f"  - Best Accuracy: {stage2_best_acc:.4f} (Epoch {stage2_best_epoch})")
    print(f"Final Evaluation Accuracy: {final_val_overall_acc:.4f}")
    print(f"{'='*80}\n")
    
    # Save experiment results to CSV if save_exp is True
    if args.save_exp:
        exp_result_file = 'exp_result.csv'
        
        # Prepare experiment record
        exp_record = {
            # Timestamp
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiment_name': args.name,
            'data_ver': args.data_ver,
            'remark': args.remark,
            
            # Parameters
            'dataset': args.dataset,
            'cnn_name': args.cnn_name,
            'num_views': args.num_views,
            'num_levels': args.num_level,
            'graph_net': args.graph_net,
            'batch_size': args.batchSize,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'pretraining': pretraining,
            'stage1_epochs': args.stage1_epochs,
            'stage2_epochs': args.stage2_epochs,
            'edge_dim': args.edge_dim,
            'att_lambda': args.att_lambda,
            'diff_topk': args.diff_topk,
            'rank_margin': args.rank_margin,
            'att_entropy': args.att_entropy,
            'freeze_epochs': args.freeze_epochs,
            'n_attn_heads': str(args.n_attn_heads),
            
            # Timing
            'total_time_sec': round(float(total_time), 2),
            'stage1_time_sec': round(float(stage1_time), 2),
            'stage2_time_sec': round(float(stage2_time), 2),
            
            # Stage results
            'stage1_best_acc': round(float(stage1_best_acc), 4),
            'stage1_best_epoch': int(stage1_best_epoch),
            'stage2_best_acc': round(float(stage2_best_acc), 4),
            'stage2_best_epoch': int(stage2_best_epoch),
            
            # Final evaluation metrics
            'final_overall_acc': round(float(final_val_overall_acc), 4),
            'final_mean_class_acc': round(float(final_val_mean_class_acc), 4),
            'final_macro_precision': round(float(final_macro_precision), 4),
            'final_macro_recall': round(float(final_macro_recall), 4),
            'final_macro_f1': round(float(final_macro_f1), 4),
            'final_micro_precision': round(float(final_micro_precision), 4),
            'final_micro_recall': round(float(final_micro_recall), 4),
            'final_micro_f1': round(float(final_micro_f1), 4),
            'final_weighted_precision': round(float(final_weighted_precision), 4),
            'final_weighted_recall': round(float(final_weighted_recall), 4),
            'final_weighted_f1': round(float(final_weighted_f1), 4),
            'final_loss': round(float(final_loss), 4),
        }
        
        # Convert to DataFrame
        exp_df = pd.DataFrame([exp_record])
        
        # Append to CSV file (create if doesn't exist)
        if os.path.exists(exp_result_file):
            # Append without header
            exp_df.to_csv(exp_result_file, mode='a', header=False, index=False)
        else:
            # Create new file with header
            exp_df.to_csv(exp_result_file, mode='w', header=True, index=False)
        
        print(f"Experiment results saved to {exp_result_file}")
