import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import math
import time
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def _pairwise_rank_hinge(logits: torch.Tensor, idx: torch.Tensor, margin: float = 0.2):
    """
    logits: [B,V], idx: [B,K] (hard top-k from current forward)
    returns scalar loss
    """
    B, V = logits.shape
    K    = idx.size(1)
    pos  = logits.gather(1, idx)                                   # [B,K]
    # mask out the positives, take hardest K negatives
    mask = torch.ones(B, V, dtype=torch.bool, device=logits.device)
    mask.scatter_(1, idx, False)
    neg_logits = logits.masked_fill(~mask, -1e9)
    m = min(K, max(1, V - K))
    _, neg_idx = torch.topk(neg_logits, m, dim=1)
    neg = logits.gather(1, neg_idx)                                 # [B,m]
    # hinge over all pos-neg pairs
    loss = F.relu(margin - pos.unsqueeze(2) + neg.unsqueeze(1)).mean()
    return loss

def _entropy_penalty(probs: torch.Tensor):
    p = probs.clamp_min(1e-9)
    H = -(p * p.log()).sum(dim=1).mean()                            # larger when diffuse
    return H


class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12, nclasses=40, stage1_epochs=0, stage2_epochs=0, args=None):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.nclasses = nclasses
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_time = 0
        self.stage2_time = 0
        self.args = args 
        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
    
    def save_best_model(self, epoch):
        """Save the best model weights to a special file"""
        self.model.save_best_model(self.log_dir, epoch)
    
    def load_best_model(self):
        """Load the best model weights"""
        return self.model.load_best_model(self.log_dir)
    def train(self, n_epochs):
        best_acc = 0
        best_epoch = 0
        i_acc = 0
        self.model.train()
        
        # Start timing
        stage_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"STARTING {self.model_name.upper()} TRAINING")
        print(f"{'='*60}")
        print(f"Number of epochs: {n_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of classes: {self.nclasses}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stage_start_time))}")
        print(f"{'='*60}\n")
        
        # Initialize stage timing
        stage1_start_time = stage_start_time if self.stage1_epochs > 0 else None
        stage2_start_time = None
        
        # Handle backbone freezing for Stage 1 (svcnn)
        # Note: If freeze_epochs > 0, freezing should already be done before optimizer creation
        # We just track the state here to know when to unfreeze
        freeze_epochs = 0
        backbone_frozen = False
        if self.model_name == 'svcnn' and self.args is not None:
            freeze_epochs = getattr(self.args, 'freeze_epochs', 0)
            # Check if backbone is already frozen (done before trainer initialization)
            if freeze_epochs > 0:
                # Check if any backbone params are frozen
                # For AlexNet/VGG, check net_1; for others, check net
                if hasattr(self.model, 'net'):
                    # ResNet/DenseNet/ViT: check net parameters
                    backbone_frozen = any(not p.requires_grad for n, p in self.model.net.named_parameters() 
                                          if ('fc' not in n and 'heads.head' not in n and 'classifier' not in n))
                elif hasattr(self.model, 'net_1'):
                    # AlexNet/VGG: check net_1 (features) parameters
                    backbone_frozen = any(not p.requires_grad for p in self.model.net_1.parameters())
                if backbone_frozen:
                    print(f"Backbone already frozen (will unfreeze after {freeze_epochs} epochs)")
        
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            
            # Track stage transitions
            if self.stage1_epochs > 0 and epoch == self.stage1_epochs:
                self.stage1_time = time.time() - stage1_start_time
                stage2_start_time = time.time()
            
            # Handle backbone unfreezing after freeze_epochs
            if self.model_name == 'svcnn' and backbone_frozen and epoch == freeze_epochs:
                if hasattr(self.model, 'unfreeze_backbone'):
                    # Get current learning rate before recreating optimizer
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.model.unfreeze_backbone()
                    backbone_frozen = False
                    print(f"\nBackbone unfrozen at epoch {epoch + 1} (starting full fine-tuning)")
                    # Recreate optimizer to include unfrozen parameters
                    if self.args is not None:
                        if hasattr(self.args, 'cnn_name') and self.args.cnn_name.startswith('vit'):
                            # ViT uses AdamW with different LRs for body and head
                            head, body = [], []
                            for n, p in self.model.net.named_parameters():
                                if 'heads.head' in n:
                                    head.append(p)
                                else:
                                    body.append(p)
                            import torch.optim as optim
                            self.optimizer = optim.AdamW([{'params': body, 'lr': 5e-5},
                                                         {'params': head, 'lr': 5e-4}], weight_decay=0.05)
                        else:
                            # ResNet/AlexNet/VGG/DenseNet uses SGD - keep current LR
                            # Include only trainable parameters (all should be trainable after unfreezing)
                            # For AlexNet/VGG, model.net doesn't exist - use model.parameters() instead
                            import torch.optim as optim
                            weight_decay = getattr(self.args, 'weight_decay', 0.001)
                            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                            if len(trainable_params) == 0:
                                raise ValueError("No trainable parameters after unfreezing! Check model state.")
                            self.optimizer = optim.SGD(trainable_params, lr=current_lr,
                                                       weight_decay=weight_decay, momentum=0.9)
            
            lr = self.optimizer.param_groups[0]['lr']
            if self.model_name == 'view_gcn':
                if epoch == 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                if epoch > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * ( 1 + math.cos(epoch * math.pi / 15))
            else:
                if epoch > 0 and (epoch + 1) % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new
            # plot learning rate
            # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)
            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                if hasattr(self.model, "update_temperatures"):
                    total_steps  = len(self.train_loader) * n_epochs
                    current_step = epoch * len(self.train_loader) + i + 1
                    # self.model.update_temperatures(current_step, total_steps)
                    tau_start       = float(getattr(self.args, 'tau_start', 2.0))
                    tau_end         = float(getattr(self.args, 'tau_end', 0.35))
                    logit_tau_start = float(getattr(self.args, 'logit_tau_start', 1.2))
                    logit_tau_end   = float(getattr(self.args, 'logit_tau_end', 0.9))
                    # self.model.update_temperatures(
                    #     step=current_step, total_steps=total_steps,
                    #     tau_start=tau_start, tau_end=tau_end,
                    #     logit_tau_start=logit_tau_start, logit_tau_end=logit_tau_end
                    # )
                    self.model.update_temperatures(step=current_step, total_steps=total_steps,
                          tau_start=3.5, tau_end=0.15,
                          logit_tau_start=1.2, logit_tau_end=0.9)
                    if (i % 50) == 0 and hasattr(self.model, 'topk1'):
                        self.writer.add_scalar('selector/tau1',      float(self.model.topk1.temperature),  current_step)
                        self.writer.add_scalar('selector/logit_tau1',float(self.model.topk1.logit_tau), current_step)
                if self.model_name == 'view_gcn' and epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))
                if self.model_name == 'view_gcn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()
                # target_ = target.unsqueeze(1).repeat(1, 4*(self.nclasses//3+5)).view(-1)
                self.optimizer.zero_grad()
                if self.model_name == 'view_gcn':
                    out_data, attention_info = self.model(in_data)

                    # (a) classification loss
                    loss_cls = self.loss_fn(out_data, target)

                    # (b) scorer regularization (rank-margin + entropy), averaged across L1–L3
                    rank_loss = torch.tensor(0., device=out_data.device)
                    ent_loss  = torch.tensor(0., device=out_data.device)

                    if isinstance(attention_info, dict) \
                    and ('topk_indices' in attention_info) \
                    and ('saliency_logits' in attention_info):

                        logits_list = attention_info['saliency_logits']            # tuple of [B,V]
                        probs_list  = attention_info.get('saliency_probs',
                                        attention_info.get('soft_masks'))       # tuple of [B,V]
                        idx_list    = attention_info['topk_indices']               # tuple of [B,K]
                        
                        # hyperparams from CLI (with safe defaults if args is None)
                        margin = float(getattr(self.args, 'rank_margin', 0.2))
                        lambda_rank = float(getattr(self.model, 'att_lambda', 0.0))      # --att_lambda to the model
                        lambda_ent  = float(getattr(self.args, 'att_entropy', 0.01))

                        for lg, pr, ix in zip(logits_list, probs_list, idx_list):
                            rank_loss = rank_loss + _pairwise_rank_hinge(lg, ix, margin=margin)
                            ent_loss  = ent_loss  + _entropy_penalty(pr)

                        n_levels  = max(1, len(idx_list))
                        rank_loss = rank_loss / n_levels
                        ent_loss  = ent_loss  / n_levels

                    # weight comes from the model (set via --att_lambda); small entropy helper
                    lambda_rank = float(getattr(self.model, 'att_lambda', 0.0))
                    lambda_ent  = 0.01

                    loss = loss_cls + lambda_rank * (rank_loss + lambda_ent * ent_loss)

                    # (optional) log components
                    self.writer.add_scalar('train/loss_cls',   loss_cls,   i_acc + i + 1)
                    self.writer.add_scalar('train/loss_rank',  rank_loss,  i_acc + i + 1)
                    self.writer.add_scalar('train/loss_ent',   ent_loss,   i_acc + i + 1)
                    # Clear attention_info to save memory (especially important for VGG16)
                    if isinstance(attention_info, dict):
                        del attention_info
                else:
                    out_data = self.model(in_data)
                    loss = self.loss_fn(out_data, target)

                
                # Check for NaN or Inf values and handle them
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf loss detected at step {i+1}, skipping this batch")
                    continue
                    
                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                #print('lr = ', str(param_group['lr']))
                loss.backward()
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                # Clear intermediate variables to save memory (especially important for VGG16)
                # Note: Don't delete variables used in log_str above
                del pred, results, correct_points
                # Periodically clear CUDA cache during training
                if (i + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i
            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    (loss, val_overall_acc, val_mean_class_acc, macro_f1, f1_scores_per_class,
                     macro_precision, macro_recall, micro_precision, micro_recall, micro_f1,
                     weighted_precision, weighted_recall, weighted_f1, precision_per_class, recall_per_class) = self.update_validation_accuracy(epoch)
                
                # Log overall metrics
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)
                
                # Log macro-averaged metrics
                self.writer.add_scalar('val/val_macro_precision', macro_precision, epoch + 1)
                self.writer.add_scalar('val/val_macro_recall', macro_recall, epoch + 1)
                self.writer.add_scalar('val/val_macro_f1', macro_f1, epoch + 1)
                
                # Log micro-averaged metrics
                self.writer.add_scalar('val/val_micro_precision', micro_precision, epoch + 1)
                self.writer.add_scalar('val/val_micro_recall', micro_recall, epoch + 1)
                self.writer.add_scalar('val/val_micro_f1', micro_f1, epoch + 1)
                
                # Log weighted-averaged metrics
                self.writer.add_scalar('val/val_weighted_precision', weighted_precision, epoch + 1)
                self.writer.add_scalar('val/val_weighted_recall', weighted_recall, epoch + 1)
                self.writer.add_scalar('val/val_weighted_f1', weighted_f1, epoch + 1)
                
                # Log per-class metrics
                for i in range(self.nclasses):
                    self.writer.add_scalar(f'val/val_precision_class_{i}', precision_per_class[i], epoch + 1)
                    self.writer.add_scalar(f'val/val_recall_class_{i}', recall_per_class[i], epoch + 1)
                    self.writer.add_scalar(f'val/val_f1_class_{i}', f1_scores_per_class[i], epoch + 1)
                
                self.model.save(self.log_dir, epoch)
                
                # save best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                    best_epoch = epoch
                    # Save the best model weights
                    self.save_best_model(epoch)
                print('best_acc', best_acc)
            
            # Calculate and print epoch timing
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"\nEpoch {epoch + 1} completed in {epoch_duration:.2f} seconds ({epoch_duration/60:.2f} minutes)")
            print(f"Average time per step: {epoch_duration/(i+1):.3f} seconds")
            print(f"Current best accuracy: {best_acc:.4f}")
            print("-" * 60)
        
        # Calculate total stage time and stage2 time
        stage_end_time = time.time()
        total_stage_time = stage_end_time - stage_start_time
        
        # Calculate stage2 time if stage1_epochs is set
        if self.stage1_epochs > 0 and stage2_start_time is not None:
            self.stage2_time = stage_end_time - stage2_start_time
        elif self.stage1_epochs > 0:
            # If stage2 never started, stage2_time is 0
            self.stage2_time = 0
        
        # Load best model for final evaluation
        print("\n" + "="*60)
        print("LOADING BEST MODEL FOR FINAL EVALUATION")
        print("="*60)
        if not self.load_best_model():
            print("Warning: Could not load best model, using final epoch model")
        
        # Generate confusion matrix for final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION - GENERATING CONFUSION MATRIX")
        print("="*60)
        cm_start_time = time.time()
        self.generate_confusion_matrix(n_epochs)
        cm_end_time = time.time()
        cm_duration = cm_end_time - cm_start_time
        
        # Print final timing summary
        print(f"\n{'='*60}")
        print(f"{self.model_name.upper()} TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"Training: {n_epochs} epochs, {total_stage_time:.2f} seconds ({total_stage_time/60:.2f} minutes)")
        print(f"{'='*60}\n")
        
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()
        
        return total_stage_time + cm_duration, best_acc, best_epoch

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(self.nclasses)
        samples_class = np.zeros(self.nclasses)
        all_loss = 0
        
        # For F1 score calculation
        all_predictions = []
        all_targets = []
        
        self.model.eval()

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'view_gcn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()
            if self.model_name == 'view_gcn':
                out_data, attention_info = self.model(in_data)
            else:
                out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            # Collect predictions and targets for F1 calculation
            all_predictions.extend(pred.cpu().data.numpy())
            all_targets.extend(target.cpu().data.numpy())

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        
        # Convert predictions and targets to numpy arrays for one-vs-rest calculation
        all_predictions_np = np.array(all_predictions)
        all_targets_np = np.array(all_targets)
        
        # Calculate one-vs-rest accuracy for each class
        class_acc = np.zeros(self.nclasses)
        for i in range(self.nclasses):
            if samples_class[i] > 0:
                # One-vs-rest: class i vs all others
                # True positives: correctly predicted as class i
                tp = np.sum((all_targets_np == i) & (all_predictions_np == i))
                # True negatives: correctly predicted as not class i
                tn = np.sum((all_targets_np != i) & (all_predictions_np != i))
                # False positives: incorrectly predicted as class i
                fp = np.sum((all_targets_np != i) & (all_predictions_np == i))
                # False negatives: incorrectly predicted as not class i
                fn = np.sum((all_targets_np == i) & (all_predictions_np != i))
                
                # One-vs-rest accuracy = (TP + TN) / (TP + TN + FP + FN)
                total_samples = tp + tn + fp + fn
                if total_samples > 0:
                    class_acc[i] = (tp + tn) / total_samples
                else:
                    class_acc[i] = 0.0
            else:
                class_acc[i] = 0.0  # Set to 0 for classes with no samples
        
        # Calculate mean accuracy only for classes that have samples
        valid_classes = samples_class > 0
        if np.any(valid_classes):
            val_mean_class_acc = np.mean(class_acc[valid_classes])
        else:
            val_mean_class_acc = 0.0
            
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Calculate overall metrics
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        macro_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
        macro_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        micro_precision = precision_score(all_targets, all_predictions, average='micro', zero_division=0)
        micro_recall = recall_score(all_targets, all_predictions, average='micro', zero_division=0)
        micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
        weighted_precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        weighted_recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        weighted_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1_scores_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)

        # Print overall metrics
        print(f"\n{'='*60}")
        print("OVERALL EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Loss: {loss:.4f}")
        print(f"\nMacro-averaged metrics:")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall: {macro_recall:.4f}")
        print(f"  F1-Score: {macro_f1:.4f}")
        print(f"\nMicro-averaged metrics:")
        print(f"  Precision: {micro_precision:.4f}")
        print(f"  Recall: {micro_recall:.4f}")
        print(f"  F1-Score: {micro_f1:.4f}")
        print(f"\nWeighted-averaged metrics:")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall: {weighted_recall:.4f}")
        print(f"  F1-Score: {weighted_f1:.4f}")
        
        # Generate and display confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Create class names
        if self.nclasses == 15:  # ScanObjectNN
            class_names = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display',
                          'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        else:  # ModelNet40 or other
            class_names = [f'Class_{i}' for i in range(self.nclasses)]
        
        # Print confusion matrix
        print(f"\n{'='*60}")
        print("CONFUSION MATRIX")
        print(f"{'='*60}")
        print("Class names:")
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")
        print("\nConfusion Matrix (Row = True Class, Column = Predicted Class):")
        print(cm)
        print(f"{'='*60}")
        
        # Print per-class metrics
        print(f"\n{'='*60}")
        print("PER-CLASS EVALUATION METRICS")
        print(f"{'='*60}")
        print(f"{'Class':<12} {'Samples':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 80)
        
        for i in range(self.nclasses):
            class_name = f"Class_{i}" if i < 10 else f"Class_{i}"
            if self.nclasses == 15:  # For ScanObjectNN, use actual class names
                class_names_list = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display',
                                   'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
                class_name = class_names_list[i] if i < len(class_names_list) else f"Class_{i}"
            elif self.nclasses == 6:  # For Colombia, use numeric class names
                class_name = str(i)
            
            samples = int(samples_class[i])
            # Use one-vs-rest accuracy calculated above
            ovr_accuracy = class_acc[i]
            precision = precision_per_class[i]
            recall = recall_per_class[i]
            f1 = f1_scores_per_class[i]
            
            print(f"{class_name:<12} {samples:<8} {ovr_accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        # Use the mean of one-vs-rest accuracies for the macro average
        mean_class_accuracy = val_mean_class_acc
        
        # Add mean (macro) values line
        print("-" * 80)
        print(f"{'MEAN (MACRO)':<12} {'':<8} {mean_class_accuracy:<10.4f} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        print(f"{'='*60}")
        self.model.train()

        return (loss, val_overall_acc, val_mean_class_acc, macro_f1, f1_scores_per_class, 
                macro_precision, macro_recall, micro_precision, micro_recall, micro_f1,
                weighted_precision, weighted_recall, weighted_f1, precision_per_class, recall_per_class)

    def generate_confusion_matrix(self, epoch, class_names=None):
        """Generate and save confusion matrix for final evaluation"""
        all_predictions = []
        all_targets = []
        self.model.eval()
        
        print("Generating confusion matrix...")
        
        with torch.no_grad():
            for _, data in enumerate(self.val_loader, 0):
                if self.model_name == 'view_gcn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:  # 'svcnn'
                    in_data = Variable(data[1]).cuda()
                target = Variable(data[0]).cuda()
                
                if self.model_name == 'view_gcn':
                    out_data, attention_info = self.model(in_data)
                else:
                    out_data = self.model(in_data)
                pred = torch.max(out_data, 1)[1]
                
                # Collect predictions and targets
                all_predictions.extend(pred.cpu().data.numpy())
                all_targets.extend(target.cpu().data.numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Create class names if not provided
        if class_names is None:
            if self.nclasses == 15:  # ScanObjectNN
                class_names = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display',
                              'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
            elif self.nclasses == 6:  # Colombia
                class_names = ['0', '1', '2', '3', '4', '5']
            else:  # ModelNet40 or other
                class_names = [f'Class_{i}' for i in range(self.nclasses)]
        
        # Create confusion matrix plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = f'{self.log_dir}/confusion_matrix_epoch_{epoch}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix as text file
        cm_text_path = f'{self.log_dir}/confusion_matrix_epoch_{epoch}.txt'
        with open(cm_text_path, 'w') as f:
            f.write(f'Confusion Matrix - Epoch {epoch}\n')
            f.write('=' * 50 + '\n\n')
            f.write('Class names:\n')
            for i, name in enumerate(class_names):
                f.write(f'{i}: {name}\n')
            f.write('\nConfusion Matrix:\n')
            f.write(str(cm))
            f.write('\n\nRow = True Class, Column = Predicted Class\n')
        
        # Print confusion matrix to console
        print(f"\nConfusion Matrix (Epoch {epoch}):")
        print("=" * 50)
        print("Class names:")
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")
        print("\nConfusion Matrix:")
        print(cm)
        print("Row = True Class, Column = Predicted Class")
        print(f"Confusion matrix saved to: {cm_path}")
        print(f"Confusion matrix text saved to: {cm_text_path}")
        
        # Calculate and print per-class metrics with one-vs-rest accuracy
        print("\nPer-class metrics:")
        print("-" * 80)
        print(f"{'Class':<12} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
        print("-" * 80)
        
        # Store metrics for mean calculation
        ovr_accuracies = []
        recalls = []
        precisions = []
        f1s = []
        
        for i in range(self.nclasses):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            true_negatives = cm.sum() - true_positives - false_positives - false_negatives
            
            # Calculate one-vs-rest accuracy
            total_samples = true_positives + true_negatives + false_positives + false_negatives
            ovr_accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
            
            # Calculate per-class metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            ovr_accuracies.append(ovr_accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            
            print(f"{class_names[i]:<12} {ovr_accuracy:<10.3f} {recall:<10.3f} {precision:<10.3f} {f1:<10.3f}")
        
        # Calculate and print mean (macro) values
        mean_ovr_accuracy = np.mean(ovr_accuracies)
        mean_recall = np.mean(recalls)
        mean_precision = np.mean(precisions)
        mean_f1 = np.mean(f1s)
        
        print("-" * 80)
        print(f"{'MEAN (MACRO)':<12} {mean_ovr_accuracy:<10.3f} {mean_recall:<10.3f} {mean_precision:<10.3f} {mean_f1:<10.3f}")
        
        self.model.train()
        return cm
