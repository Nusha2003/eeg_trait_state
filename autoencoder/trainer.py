
import os, shutil, tempfile, tqdm
import time 
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

import numpy as np

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# sklearn metrics
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, top_k_accuracy_score, cohen_kappa_score, roc_auc_score


from functools import partial

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize


class PreTrainerEEG:
    def __init__(
            self,
            min_delta,
            model=None,
            model_checkpoint: str = None,
            logdir: str = None,
            device: int = 0,
            is_main_process: bool=True,
            optimizer_args=None,
            scheduler_args=None,
            save_n_iters=None,
            val_n_iters=150, # number of iterations of validation set to run per evaluation
            patience_metric="loss",
            early_stop_patience=None,
            gradient_clipping: float=None,
            gradient_accumulation: int=None
    ):
        
        self.device = torch.device(f"cuda:{device}") if isinstance(device, int) else device
        self.model = model
        self.model.to(self.device)
        self.is_main_process = is_main_process
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation = gradient_accumulation if gradient_accumulation is not None else 1 

        self._accum_step = 0 
        
        self.logdir = logdir

        if self.is_main_process:
            os.makedirs(logdir, exist_ok=True)

            # Initialize tensorboard
            self.writer = SummaryWriter(log_dir=logdir)

        # History only tracks loss now
        self.history = {
            "train_loss_iter": [], 
            "train_iters": [],
            "eval_loss": [], 
            "eval_iters": []
        }
        
        self.running_train_loss = 0.0
        self.running_train_steps = 0
        self.val_n_iters = val_n_iters
        
        self.n_iters = 0
        self.save_n_iters = save_n_iters
        self.min_delta = min_delta

        # Optimizer Setup
        if optimizer_args["type"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_args["params"])
        elif optimizer_args["type"] == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_args["params"])

        self.scheduler_args = scheduler_args
        self.scheduler = self._get_scheduler(scheduler_args) if scheduler_args else None

        if model_checkpoint is not None:
            self._load_model(model_checkpoint)

        self.early_stop_counter = 0
        self.early_stop_patience = early_stop_patience
        # In pretraining, lower loss is always better
        self.patience_metric = "loss" 
        self.best_checkpoint_path = os.path.join(logdir, "best_model.pt")
        self.best_val = float('inf')
        self.current_epoch = 0



    def _get_scheduler_type(self, scheduler_args: dict = None):
        if scheduler_args["type"] == "expLR":
            return ExponentialLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "linearLR":
            return LinearLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "cosineAnnealLR":
            return CosineAnnealingLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "reducelronplateau":
            return ReduceLROnPlateau(self.optimizer, **scheduler_args["params"])

    def _get_scheduler(self, scheduler_args):
        if scheduler_args is None: return None
        if isinstance(scheduler_args, dict):
            return self._get_scheduler_type(scheduler_args)
        elif isinstance(scheduler_args, list):
            list_schedulers, milestones, current_milestone = [], [], 0
            for i, s_args in enumerate(scheduler_args):
                if s_args["type"] == "reducelronplateau": 
                    raise ValueError("ReduceLROnPlateau cannot be in SequentialLR chain.")
                list_schedulers.append(self._get_scheduler_type(s_args))
                if i < len(scheduler_args) - 1:
                    current_milestone += s_args["params"]["total_iters"]
                    milestones.append(current_milestone)
            return SequentialLR(self.optimizer, schedulers=list_schedulers, milestones=milestones)
        raise NotImplementedError

    def train(self, n_iters, train_dataloader=None, val_dataloader=None, **kwargs):
        """
        Train for a specific number of iterations (batches), disregarding epochs.
        """
        if self.is_main_process:
            print(f"Starting Pretraining for {n_iters} iterations...")
            pbar = tqdm.tqdm(total=n_iters, initial=self.n_iters, desc="Training Steps")
        else:
            pbar = None

        if self.save_n_iters is None:
            self.save_n_iters = 1000

        data_start_time = time.time()

        # Create an infinite iterator over the dataloader
        train_iterator = iter(train_dataloader)

        while self.n_iters < n_iters:
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Dataset exhausted its 'samples_per_epoch': restart iterator
                self.current_epoch += 1
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            data_time = time.time() - data_start_time

            compute_start_time = time.time()

            # --- Training Step ---
            step_loss, did_optmiize_step = self._train_one_step(batch, **kwargs)
            
            compute_time = time.time() - compute_start_time

            if did_optmiize_step:
                if self.is_main_process and self.n_iters % 50 == 0:
                    self.writer.add_scalar("Perf/Data_IO_Time", data_time, self.n_iters)
                    self.writer.add_scalar("Perf/Compute_Time", compute_time, self.n_iters)
                    
                    # Log Peak GPU Memory Allocation in MB
                    mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                    self.writer.add_scalar("Perf/Peak_GPU_Memory_MB", mem_mb, self.n_iters)

                if self.is_main_process:
                    pbar.set_postfix({"Loss": f"{step_loss:.4f}"})
                    pbar.update(1)
                
                if self.n_iters % self.save_n_iters == 0 and self.n_iters > 0:
                    self._evaluate(val_dataloader)
                    
                    # Scheduler Step (Plateau)
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(self.history["eval_loss"][-1])

                    self.model.train()
                    
                    # Log Training Loss
                    if self.is_main_process:
                        avg_train_loss = self.running_train_loss / self.running_train_steps
                        self.history["train_loss_iter"].append(avg_train_loss)
                        self.history["train_iters"].append(self.n_iters)
                        self.writer.add_scalar("Loss/Train_Iter", avg_train_loss, self.n_iters)
                        
                    # Reset Running Counters
                    self.running_train_loss = 0.0
                    self.running_train_steps = 0
                    if self.is_main_process:
                        self.writer.flush()
                
                    # Check Early Stopping
                    if self.early_stop_patience and self.early_stop_counter >= self.early_stop_patience:
                        print(f"Early stopping triggered at iteration {self.n_iters}")
                        break

                # Step Scheduler (Cyclic/Sequential)
                if not isinstance(self.scheduler, ReduceLROnPlateau) and self.scheduler is not None:
                    self.scheduler.step()
                
                data_start_time = time.time()
  
        if self.is_main_process:
            pbar.close()
            self.writer.close()

        return self.best_val

    def _train_one_step(self, batch, **kwargs):
        """
        Processes a single batch.
        """
        self.model.train()
        
        # Move dictionary tensors to device
        batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

        is_first_micro_batch = (self._accum_step == 0)
        if is_first_micro_batch:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Unpack batch dictionary as kwargs for the model
        # outputs = self.model(**batch, **kwargs)
        # loss = outputs["loss"]
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # The model automatically casts linear layers, attentions, etc. to BF16
            outputs = self.model(**batch, **kwargs)
            loss = outputs["loss"]/self.gradient_accumulation
        
        loss.backward()
        self._accum_step += 1

        unscaled_loss = loss.detach() * self.gradient_accumulation

        did_optimizer_step = False
        if self._accum_step >= self.gradient_accumulation:
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
            
            self.optimizer.step()
            if hasattr(self.model, "update_target_network"):
                self.model.update_target_network()
            self._accum_step = 0 
            did_optimizer_step = True
            self.n_iters += 1   

        loss_val_tensor = unscaled_loss.clone()
        if dist.is_initialized():
            dist.all_reduce(loss_val_tensor, op=dist.ReduceOp.AVG)
        loss_val = loss_val_tensor.item()
        
        self.running_train_loss += loss_val/self.gradient_accumulation
        self.running_train_steps += 1
        
        return loss_val, did_optimizer_step

    def _save_model(self, checkpoint_path: str = None):
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        torch.save({
            "n_iters": self.n_iters, 
            "history": self.history, 
            "model_state_dict": model_to_save.state_dict(), 
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None, 
            "best_val": self.best_val,                            # Added
            "early_stop_counter": self.early_stop_counter         # Added
        }, checkpoint_path)

    def _load_model(self, checkpoint_path: str = None):
        info = torch.load(checkpoint_path, map_location=self.device)
        self.n_iters = info["n_iters"]
        self.history = info["history"]
        self.best_val = info.get("best_val", float('inf'))        
        self.early_stop_counter = info.get("early_stop_counter", 0) 

        # Corrected: Only load the state dict ONCE into the underlying module
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(info["model_state_dict"])
        
        self.optimizer.load_state_dict(info["optimizer_state_dict"])
        if self.scheduler and info["scheduler_state_dict"]: 
            self.scheduler.load_state_dict(info["scheduler_state_dict"])
    
    @torch.no_grad()
    def _evaluate(self, dataloader):
        """
        Evaluates reconstruction loss on the validation set.
        """
        if dataloader is None:
            return

        self.model.eval()
        
        # Initialize as strictly float32 tensors to prevent DDP type mismatch crashes
        eval_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        total_batches = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        for batch in dataloader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(**batch)
            # outputs = self.model(**batch)
            
            eval_loss += outputs["loss"].detach().to(torch.float32)
            del outputs
            
            total_batches += 1.0

            if total_batches >= self.val_n_iters:
                break
        
        if dist.is_initialized():
            dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)

        # Calculate average safely
        avg_eval_loss = (eval_loss / total_batches).item() if total_batches.item() > 0 else 0.0

        if avg_eval_loss < self.best_val - self.min_delta:
            self.best_val = avg_eval_loss
            self.early_stop_counter = 0 
            is_new_best = True
        else:
            self.early_stop_counter += 1
            is_new_best = False

        if self.is_main_process:
            self.history["eval_loss"].append(avg_eval_loss)
            self.history["eval_iters"].append(self.n_iters)
            self.writer.add_scalar("Loss/Eval", avg_eval_loss, self.n_iters)

            print(f"\n[Eval @ Iter {self.n_iters}] Synced Loss: {avg_eval_loss:.4f}")
            
            self._save_model(os.path.join(self.logdir, f"checkpoint_{self.n_iters}.pt"))
            if is_new_best:
                self._save_model(self.best_checkpoint_path)
                print(f"New best model saved with loss {self.best_val:.4f}")
    
    @torch.no_grad()
    def save_embeddings(
        self,
        dataloader,
        save_path,
        split="test",
    ):
        self._load_model(
            self.best_checkpoint_path
        )

        self.model.eval()

        os.makedirs(
            save_path,
            exist_ok=True,
        )

        all_embeddings = []

        # Common metadata
        all_subjects = []
        all_conditions = []

        # Optional numeric metadata
        all_runs = []
        all_epochs = []
        all_trials = []
        all_original_indices = []
        all_sfreqs = []
        all_moabb_indices = []

        # Optional string metadata
        all_files = []
        all_original_parts = []
        all_sessions = []

        for batch in dataloader:
            if not isinstance(batch, dict):
                raise TypeError(
                    "Expected each dataloader batch "
                    "to be a dictionary."
                )

            inputs = batch["x"].to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(
                x=inputs,
                labels=None,
                return_embeddings=True,
            )

            z = outputs["embeddings"]

            # Convert latent output to:
            # (batch_size, latent_dim)
            z = torch.flatten(
                z,
                start_dim=1,
            )

            all_embeddings.append(
                z.detach()
                .cpu()
                .numpy()
            )

            # ==========================================
            # Common metadata
            # ==========================================

            if "subject" in batch:
                all_subjects.append(
                    batch["subject"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            if "condition" in batch:
                all_conditions.append(
                    batch["condition"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            # ==========================================
            # Optional numeric metadata
            # ==========================================

            if "run" in batch:
                if torch.is_tensor(
                    batch["run"]
                ):
                    all_runs.append(
                        batch["run"]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    all_runs.extend(
                        list(batch["run"])
                    )

            if "epoch" in batch:
                all_epochs.append(
                    batch["epoch"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            if "trial" in batch:
                all_trials.append(
                    batch["trial"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            if "original_index" in batch:
                all_original_indices.append(
                    batch["original_index"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            if "sfreq" in batch:
                all_sfreqs.append(
                    batch["sfreq"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            if "moabb_index" in batch:
                all_moabb_indices.append(
                    batch["moabb_index"]
                    .detach()
                    .cpu()
                    .numpy()
                )

            # ==========================================
            # Optional string metadata
            # ==========================================

            if "file" in batch:
                all_files.extend(
                    list(batch["file"])
                )

            if "original_part" in batch:
                all_original_parts.extend(
                    list(
                        batch["original_part"]
                    )
                )

            if "session" in batch:
                all_sessions.extend(
                    list(batch["session"])
                )

        if len(all_embeddings) == 0:
            raise ValueError(
                "The dataloader produced no batches."
            )

        embeddings = np.concatenate(
            all_embeddings,
            axis=0,
        )

        n_samples = embeddings.shape[0]

        # ==========================================
        # Save embeddings
        # ==========================================

        np.save(
            os.path.join(
                save_path,
                f"{split}_embeddings.npy",
            ),
            embeddings,
        )

        print(
            f"\nSaved {split} embeddings"
        )
        print(
            f"Embeddings: {embeddings.shape}"
        )

        # ==========================================
        # Helper for optional numeric arrays
        # ==========================================

        def save_numeric(
            values,
            name,
        ):
            if not values:
                return None

            if isinstance(
                values[0],
                np.ndarray,
            ):
                array = np.concatenate(
                    values,
                    axis=0,
                )
            else:
                array = np.asarray(
                    values
                )

            if len(array) != n_samples:
                raise ValueError(
                    f"{name} length does not "
                    f"match embeddings: "
                    f"{len(array)} vs {n_samples}"
                )

            np.save(
                os.path.join(
                    save_path,
                    f"{split}_{name}.npy",
                ),
                array,
            )

            print(
                f"{name}: {array.shape}"
            )

            return array

        # ==========================================
        # Save common metadata
        # ==========================================

        subjects = save_numeric(
            all_subjects,
            "subjects",
        )

        conditions = save_numeric(
            all_conditions,
            "conditions",
        )

        # ==========================================
        # Save dataset-specific numeric metadata
        # ==========================================

        runs = save_numeric(
            all_runs,
            "runs",
        )

        epochs = save_numeric(
            all_epochs,
            "epochs",
        )

        trials = save_numeric(
            all_trials,
            "trials",
        )

        original_indices = save_numeric(
            all_original_indices,
            "original_indices",
        )

        sfreqs = save_numeric(
            all_sfreqs,
            "sfreqs",
        )

        moabb_indices = save_numeric(
            all_moabb_indices,
            "moabb_indices",
        )

        # ==========================================
        # Helper for optional string arrays
        # ==========================================

        def save_strings(
            values,
            name,
        ):
            if not values:
                return None

            array = np.asarray(
                values,
                dtype=str,
            )

            if len(array) != n_samples:
                raise ValueError(
                    f"{name} length does not "
                    f"match embeddings: "
                    f"{len(array)} vs {n_samples}"
                )

            np.save(
                os.path.join(
                    save_path,
                    f"{split}_{name}.npy",
                ),
                array,
            )

            print(
                f"{name}: {array.shape}"
            )

            return array

        # ==========================================
        # Save dataset-specific string metadata
        # ==========================================

        files = save_strings(
            all_files,
            "files",
        )

        original_parts = save_strings(
            all_original_parts,
            "original_parts",
        )

        sessions = save_strings(
            all_sessions,
            "sessions",
        )

        # ==========================================
        # Summary
        # ==========================================

        if subjects is not None:
            print(
                "Unique subjects:",
                np.unique(subjects).size,
            )

        if conditions is not None:
            print(
                "Unique conditions:",
                np.unique(conditions),
            )

        if runs is not None:
            print(
                "Unique runs:",
                np.unique(runs),
            )

        if trials is not None:
            print(
                "Unique trials:",
                np.unique(trials).size,
            )
"""
class TrainerEEG:
    def __init__(
            self,
            model=None,
            model_checkpoint: str=None,
            logdir: str=None,
            device: str="cuda",
            optimizer_args=None,
            scheduler_args=None,
            save_n_iters=None,
            save_n_epochs=None,
            patience_metric="bac",
            early_stop_patience=3,
            metrics: Union[list, str]=None,
            gradient_clipping: float=None,
            input_chans: list=None,
            model_type: str=None,
            **kwargs
            ):
        
        self.input_chans = input_chans # used in LaBram

        self.model = model.to(device)
        self.device = device
        self.metrics = metrics if metrics else []
        self.gradient_clipping = gradient_clipping
        self.model_type = model_type
        
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=logdir)

        self.history = {
            "train_loss_iter": [], 
            "train_iters": [],
            "train_loss_epoch": [],
            "eval_loss": [], 
            "eval_iters": []
        }
        
        self.running_train_loss = 0.0
        self.running_train_steps = 0
        
        self.test_loss, self.test_metrics = 0, dict()
        self.train_metrics, self.eval_metrics = dict(), dict()

        # Metric parsing
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]
        
        for m in self.metrics:
            self.eval_metrics[m] = list()

        self.n_iters, self.n_epoch = 0, 0
        self.save_n_iters, self.save_n_epochs = save_n_iters, save_n_epochs

        params_dict = optimizer_args["params"].copy()

        if "labram" in self.model_type:
            # labram has layer wise lr decay
            lr = params_dict.pop("lr", 1e-4)
            weight_decay = params_dict.pop("weight_decay", 0.0)
            optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay, lr=lr)
            for i, group in enumerate(optim_groups):
                print(f"Group {i}: LR={group['lr']:.2e}, WD={group['weight_decay']:.2e}, Params={len(group['params'])}")
        else:
            weight_decay = params_dict.pop("weight_decay", 0.0)
            optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay)

        # optim_groups = self.model.get_optimizer_params(weight_decay=weight_decay)
        # Optimizer Setup
        if optimizer_args["type"] == "adam":
            self.optimizer = torch.optim.Adam(optim_groups, **params_dict)
        elif optimizer_args["type"] == "adamw":
            self.optimizer = torch.optim.AdamW(optim_groups, **params_dict)

        self.scheduler_args = scheduler_args
        self.scheduler = self._get_scheduler(scheduler_args) if scheduler_args else None

        if model_checkpoint is not None:
            self._load_model(model_checkpoint)

        self.early_stop_counter = 0
        self.early_stop_patience = early_stop_patience
        self.patience_metric = "loss" if patience_metric is None else patience_metric
        self.best_checkpoint_path = os.path.join(logdir, "best_model.pt")
        self.best_val = -float('inf') if self.patience_metric != "loss" else float('inf')

    def _get_scheduler_type(self, scheduler_args: dict=None):
        if scheduler_args["type"] == "expLR":
            return ExponentialLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "linearLR":
            return LinearLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "cosineAnnealLR":
            return CosineAnnealingLR(self.optimizer, **scheduler_args["params"])
        elif scheduler_args["type"] == "reducelronplateau":
            return ReduceLROnPlateau(self.optimizer, **scheduler_args["params"])

    def _get_scheduler(self, scheduler_args: Union[list, dict] = None):
        if scheduler_args is None: return None
        if isinstance(scheduler_args, dict):
            return self._get_scheduler_type(scheduler_args)
        elif isinstance(scheduler_args, list):
            list_schedulers, milestones, current_milestone = [], [], 0
            for i, s_args in enumerate(scheduler_args):
                if s_args["type"] == "reducelronplateau": raise ValueError("ReduceLROnPlateau cannot be in SequentialLR chain.")
                list_schedulers.append(self._get_scheduler_type(s_args))
                if i < len(scheduler_args) - 1:
                    current_milestone += s_args["params"]["total_iters"]
                    milestones.append(current_milestone)
            return SequentialLR(self.optimizer, schedulers=list_schedulers, milestones=milestones)
        raise NotImplementedError

    def get_iterator(self, start_epoch, n_epochs):
        return tqdm.tqdm(range(start_epoch, n_epochs), desc="Epochs")
    
    def train(self, n_epochs, n_iters=None, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        if n_epochs is None and n_iters is not None:
            n_epochs = int(n_iters // len(train_dataloader))
        
        # Default save frequency if not provided
        if self.save_n_iters is None:
            if self.save_n_epochs is not None:
                self.save_n_iters = self.save_n_epochs * len(train_dataloader)
            else:
                self.save_n_iters = len(train_dataloader) # Default to 1 epoch

        start_epoch = self.n_epoch
        iterator = self.get_iterator(start_epoch, n_epochs)

        for n in iterator:
            epoch_loss = self._train_one_epoch(train_dataloader, val_dataloader)
            
            # Log Epoch Loss
            self.history["train_loss_epoch"].append(epoch_loss)
            self.writer.add_scalar("Loss/Train_Epoch", epoch_loss, n)
            
            # Update progress bar
            # iterator.set_postfix({"Epoch Loss": f"{epoch_loss:.4f}"})
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"Epoch Loss": f"{epoch_loss:.4f}"})

            self.n_epoch = n + 1

            if hasattr(self, "early_stop_counter") and self.early_stop_patience and self.early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {n}")
                break

        return self.best_val

    def _train_one_epoch(self, train_dataloader, val_dataloader):
        self.model.train()
        
        epoch_total_loss = 0.0
        epoch_total_samples = 0

        for i, batch in enumerate(train_dataloader):
            adv_labels=None
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels", None)
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)
            else:
                inputs, labels = batch
                channel_ids = None

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            self.optimizer.zero_grad()
            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs) 
            loss = outputs["loss"]
            
            loss.backward()

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping)
            self.optimizer.step()

            loss_val = loss.item()
            batch_size = inputs.size(0)

            self.running_train_loss += loss_val
            self.running_train_steps += 1

            epoch_total_loss += loss_val
            epoch_total_samples += 1

            # Log Iteration Loss
            if self.n_iters % self.save_n_iters == 0 and self.n_iters > 0:
                self._evaluate_classification(val_dataloader)
                
                # Scheduler Step (Plateau)
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(self.history["eval_loss"][-1])

                self.model.train()
                
                avg_iter_loss = self.running_train_loss / self.running_train_steps
                
                self.history["train_loss_iter"].append(avg_iter_loss)
                self.history["train_iters"].append(self.n_iters)
                self.writer.add_scalar("Loss/Train_Iter", avg_iter_loss, self.n_iters)
                
                # Reset Running Counters
                self.running_train_loss = 0.0
                self.running_train_steps = 0
                self.writer.flush()

            self.n_iters += 1

        if not isinstance(self.scheduler, ReduceLROnPlateau) and self.scheduler is not None:
            self.scheduler.step()
            
        # Return the average loss for this epoch
        return epoch_total_loss / epoch_total_samples

    def _save_model(self, checkpoint_path: str=None):
        torch.save({
            "epoch": self.n_epoch, 
            "n_iters": self.n_iters, 
            "history": self.history, # Saved the unified history dict
            "eval_metrics": self.eval_metrics,
            "model_state_dict": self.model.state_dict(), 
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else "", 
            "test_metrics": self.test_metrics
        }, checkpoint_path)

    def _load_model(self, checkpoint_path: str=None):
        info = torch.load(checkpoint_path)
        self.n_epoch = info["epoch"]
        self.n_iters = info["n_iters"]
        
        if "history" in info:
            self.history = info["history"]
        else:
            self.history["train_loss_iter"] = info.get("train_loss", {}).get("loss", [])
            self.history["eval_loss"] = info.get("eval_loss", {}).get("loss", [])

        self.eval_metrics = info["eval_metrics"]
        self.test_metrics = info["test_metrics"]
        self.model.load_state_dict(info["model_state_dict"])
        self.optimizer.load_state_dict(info["optimizer_state_dict"])
        if self.scheduler: self.scheduler.load_state_dict(info["scheduler_state_dict"])

    @torch.no_grad()
    def _evaluate_classification(self, dataloader, split="val"):
        self.model.eval()
        eval_loss, total_samples = 0.0, 0
        preds, gts = [], []
        all_probs = []

        for batch in dataloader:
            adv_labels = None
            # inputs, labels = batch
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels")
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)
            else:
                inputs, labels = batch
                channel_ids = None

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs)
            
            # Normalize loss aggregation
            eval_loss += outputs["loss"].item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            if len(self.metrics) > 0:
                preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy().reshape(-1))
                gts.extend(labels.cpu().numpy().reshape(-1))

                probs = torch.softmax(outputs["logits"], dim=1)
                all_probs.extend(probs.cpu().detach().numpy())

        avg_eval_loss = eval_loss / total_samples

        if split == "val":
            self.history["eval_loss"].append(avg_eval_loss)
            self.history["eval_iters"].append(self.n_iters)
            self.writer.add_scalar("Loss/Eval", avg_eval_loss, self.n_iters)

        current_metrics = {}
        for metric in self.metrics:
            # Check if METRICS is defined in context, otherwise handle safely
            if metric in METRICS:
                if metric == "auroc":
                    prob_array = np.array(all_probs)
                    gt_array = np.array(gts)

                    if prob_array.shape[1] == 2:
                        val = METRICS[metric](gt_array, prob_array[:, 1])
                    else:
                        val = METRICS[metric](gt_array, prob_array)
                else:
                    val = METRICS[metric](np.array(gts), np.array(preds))
                
                if split == "val": 
                    self.eval_metrics[metric].append(val)
                else: 
                    self.test_metrics[metric] = val
                self.writer.add_scalar(f"Metrics/{split}/{metric}", val, self.n_iters)
                current_metrics[metric] = val

        if split == "val":
            print(f"[Eval @ Iter {self.n_iters}] Loss: {avg_eval_loss:.4f}, {current_metrics}")
            
            # Save standard checkpoint
            self._save_model(os.path.join(self.logdir, f"checkpoint_{self.n_iters}.pt"))

            # Logic to update "best_model.pt"
            target_val = current_metrics[self.patience_metric] if self.patience_metric != "loss" else avg_eval_loss
            # Flip logic for loss (lower is better) vs metrics (higher is better)
            if self.patience_metric == "loss":
                is_better = target_val < self.best_val
            else:
                is_better = target_val > self.best_val
                
            if is_better:
                self.best_val = target_val
                self.early_stop_counter = 0 # Reset patience
                self._save_model(self.best_checkpoint_path)
            else:
                self.early_stop_counter += 1
    
    def test_evaluate_classification(self, dataloader):
        self._load_model(self.best_checkpoint_path)
        self.model.eval()

        eval_loss, total_samples = 0.0, 0
        preds, gts = [], []
        all_probs = []

        for batch in dataloader:
            adv_labels = None
            channel_ids = None
            if type(batch) == dict:
                inputs = batch.pop("inputs")
                labels = batch.pop("labels")
                channel_ids = batch.pop("channel_ids", None)
                adv_labels = batch.pop("adv_labels", None)

            else:
                inputs, labels = batch

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            forward_kwargs = {"labels": labels}
            if channel_ids is not None:
                # Only add these if the dataset provided them (your new model)
                channel_ids = channel_ids.to(self.device)
                forward_kwargs["channel_ids"] = channel_ids
                forward_kwargs["padding_mask"] = None
            if adv_labels is not None:
                adv_labels = adv_labels.to(self.device)
                forward_kwargs["adv_labels"] = adv_labels

            outputs = self.model(inputs, input_chans=self.input_chans, **forward_kwargs)
            
            # Normalize loss aggregation
            eval_loss += outputs["loss"].item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            if len(self.metrics) > 0:
                preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy().reshape(-1))
                gts.extend(labels.cpu().numpy().reshape(-1))

                probs = torch.softmax(outputs["logits"], dim=1)
                all_probs.extend(probs.cpu().detach().numpy())

        avg_eval_loss = eval_loss / total_samples

        current_metrics = {}
        for metric in self.metrics:
            # Check if METRICS is defined in context, otherwise handle safely
            if metric in METRICS:
                if metric == "auroc":
                    prob_array = np.array(all_probs)
                    gt_array = np.array(gts)

                    if prob_array.shape[1] == 2:
                        val = METRICS[metric](gt_array, prob_array[:, 1])
                    else:
                        val = METRICS[metric](gt_array, prob_array)
                else:
                    val = METRICS[metric](np.array(gts), np.array(preds))
                current_metrics[metric] = val
        
        from collections import Counter
        print(f"Test Dataset label split: {Counter(gts)}")
        print(f"[Test @ Iter {self.n_iters}] Loss: {avg_eval_loss:.4f}, {current_metrics}")
        return current_metrics
"""