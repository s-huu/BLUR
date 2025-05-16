import torch
from transformers import Trainer
from utils import get_batch_loss
import copy
import torch.nn.functional as F

class DistillKL(torch.nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_s = p_s.view(-1, p_s.size(-1))
        p_t = F.softmax(y_t / self.T, dim=1)
        p_t = p_t.view(-1, p_t.size(-1))
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2) / y_s.shape[0]
        return loss

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.oracle_model = kwargs.pop('oracle_model').cuda()
        self.loss_type = kwargs.pop('forget_loss')

        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        # model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
    
class CustomTrainerGAAndNPO(CustomTrainer):
    def __init__(self, *args, **kwargs):
        self.with_kl = kwargs.pop('with_kl')

        super(CustomTrainerGAAndNPO, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, retain_inputs = inputs
        input_ids, labels, attention_mask = forget_inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        if self.loss_type == 'grad_ascent':
            loss = -1. * outputs.loss
        elif self.loss_type == 'npo':
            self.beta = 0.03
            forget_loss_current = outputs.loss
            # forget_loss_current = get_batch_loss(outputs.logits, labels)

            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
                # forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                forget_loss_oracle = forget_outputs_oracle.loss
            neg_log_ratios = forget_loss_current - forget_loss_oracle

            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
        else:
            raise Exception("Loss type not supported for CustomTrainerGAAndNPO. Please choose a different trainer.")
        if self.with_kl:
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            retain_loss = F.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss += retain_loss
        return (loss, outputs) if return_outputs else loss
    
class CustomTrainerScrub(CustomTrainer):
    def __init__(self, eps1, eps2, scrub_max_epochs=4, T=4.,*args, **kwargs):
        self.T = T
        self.epsilon1 = eps1
        self.epsilon2 = eps2
        self.distill = DistillKL(T = T)
        self.scrub_max_epochs = scrub_max_epochs
        self.previous_step = None

        super(CustomTrainerScrub, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        alpha = self.epsilon1
        gamma = self.epsilon2
        current_epoch = self.state.epoch
    
        forget_inputs, retain_inputs = inputs
        input_ids, labels, attention_mask = forget_inputs
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        # print(model.device, self.oracle_model.device)
        
        current_step = (
            "max"
            if current_epoch < self.scrub_max_epochs and int(current_epoch) % 2 == 0
            else "min"
        )

        if self.previous_step != current_step:
            self.previous_step = current_step
            print(f"SCRUB {current_step} step: {current_epoch}")
        # print(current_epoch, self.scrub_max_epochs)

    
        # Max step    
        if current_epoch < self.scrub_max_epochs and int(current_epoch) % 2 == 0:
            with torch.no_grad():
                orcale_forget_outputs = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_distill_loss = (
                self.distill(forget_outputs.logits, orcale_forget_outputs.logits)
                * -1
            )
            loss = forget_distill_loss
            outputs = forget_outputs
        # Min step
        else:
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            with torch.no_grad():
                orcale_retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)

            retain_distill_loss = self.distill(
                retain_outputs.logits, orcale_retain_outputs.logits
            )
            loss = alpha * retain_distill_loss + gamma * retain_outputs.loss
            outputs = retain_outputs

        return (loss, outputs) if return_outputs else loss

    
