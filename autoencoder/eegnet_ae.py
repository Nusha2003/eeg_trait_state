import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


class EEGNetAutoEncoder(BaseModel):
    def __init__(
        self,
        F1: int = 8,
        D: int = 2,
        F2: int = None,
        sampling_rate: int = 128,
        n_seconds_input: int = 5,
        n_channels: int = None,
        n_classes: int = None,
        p_dropout: float = 0.25,
        conv_constraint: bool = False,
        mode: str = "reconstruction",
        pretrained_checkpoint: str = None,
        finetune_type: str = "all",
        is_adversarial_training: bool = False,
        adv_num_classes: int = 2,
        adv_lambda: float = 0.5,
        **kwargs
    ):
        super(EEGNetAutoEncoder, self).__init__()

        self.n_classes = n_classes
        self.n_timesteps = int(sampling_rate * n_seconds_input)
        self.n_channels = n_channels
        self.F1 = F1
        self.D = D
        self.F2 = self.F1 * self.D if F2 is None else F2
        self.mode = mode
        self.finetune_type = finetune_type

        self.is_adversarial_training = is_adversarial_training
        self.adv_num_classes = adv_num_classes
        self.adv_lambda = adv_lambda

        # ==========================================
        #                 ENCODER
        # ==========================================
        self.conv_temp = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, sampling_rate // 2),
            padding="same"
        )
        self.batch_norm_temp = nn.BatchNorm2d(num_features=self.F1)

        if conv_constraint:
            self.conv_depthwise = Conv2dWithConstraint(
                F1, F1 * D, (n_channels, 1),
                groups=F1, padding="valid", bias=False,
                max_norm=1.0
            )
        else:
            self.conv_depthwise = nn.Conv2d(
                in_channels=self.F1,
                out_channels=self.F1 * self.D,
                kernel_size=(self.n_channels, 1),
                padding="valid",
                groups=self.F1
            )

        self.batch_norm_depthwise = nn.BatchNorm2d(num_features=self.F1 * self.D)
        self.activation = nn.ELU()
        self.avgpool_depthwise = nn.AvgPool2d(kernel_size=(1, 4), padding=(0, 0))
        self.dropout = nn.Dropout(p=p_dropout)

        self.conv_seperable = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F1 * self.D,
            kernel_size=(1, sampling_rate // 8),
            padding="same",
            groups=self.F1 * self.D
        )
        self.conv_pointwise = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F2,
            kernel_size=1,
            padding="same"
        )
        self.batch_norm_seperable = nn.BatchNorm2d(num_features=self.F2)
        self.avgpool_seperable = nn.AvgPool2d(kernel_size=(1, 8), padding=(0, 0))

        # ==========================================
        #             CLASSIFICATION HEAD
        # ==========================================
        # Only create this when classification mode is actually used
        self.flatten = nn.Flatten()
        if self.mode == "classification":
            self.fc = nn.LazyLinear(n_classes)
        else:
            self.fc = None

        # ==========================================
        #         OPTIONAL ADVERSARIAL HEAD
        # ==========================================
        # This is separate from fc_head and operates on embeddings
        if self.is_adversarial_training:
            self.adv_head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.Dropout(p_dropout),
                nn.Linear(128, adv_num_classes)
            )
        else:
            self.adv_head = None

        # ==========================================
        #                 DECODER
        # ==========================================
        self.deconv_pointwise = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.F1 * self.D,
            kernel_size=1,
            padding="same"
        )
        self.deconv_seperable = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F1 * self.D,
            kernel_size=(1, sampling_rate // 8),
            padding="same",
            groups=self.F1 * self.D
        )
        self.batch_norm_dec_sep = nn.BatchNorm2d(num_features=self.F1 * self.D)

        self.deconv_depthwise = nn.ConvTranspose2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F1,
            kernel_size=(self.n_channels, 1),
            padding=0,
            groups=self.F1
        )
        self.batch_norm_dec_depth = nn.BatchNorm2d(num_features=self.F1)

        self.deconv_temp = nn.Conv2d(
            in_channels=self.F1,
            out_channels=1,
            kernel_size=(1, sampling_rate // 2),
            padding="same"
        )

        if pretrained_checkpoint is not None:
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                else:
                    pretrained_dict = checkpoint
            else:
                pretrained_dict = checkpoint

            model_dict = self.state_dict()

            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape and not k.startswith('fc')
            }

            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} layers from {pretrained_checkpoint}")

        if self.finetune_type == "frozen":
            print("Finetune type is 'frozen'. Freezing all encoder and decoder weights...")
            for param in self.parameters():
                param.requires_grad = False

            if self.fc is not None:
                for param in self.fc.parameters():
                    param.requires_grad = True

            print("Done. Only the classification head (if present) requires gradients.")

    def forward(self, x, labels=None, adv_labels=None, return_embeddings: bool = False, **kwargs):
        loss = None
        recon_loss = None
        adv_loss = None
        adv_logits = None

        original_length = x.shape[2]
        x = torch.unsqueeze(x, 1)  # (B, 1, C, T)

        # Encoder
        x = self.conv_temp(x)
        x = self.batch_norm_temp(x)
        x = self.conv_depthwise(x)
        x = self.batch_norm_depthwise(x)
        x = self.activation(x)

        shape_before_pool1 = x.shape
        x = self.avgpool_depthwise(x)
        x = self.dropout(x)

        x = self.conv_seperable(x)
        x = self.conv_pointwise(x)
        x = self.batch_norm_seperable(x)
        x = self.activation(x)

        shape_before_pool2 = x.shape
        x = self.avgpool_seperable(x)
        embeddings = self.dropout(x)

        # Route by mode
        if self.mode == "classification":
            if self.fc is None:
                raise RuntimeError("Classification mode requested, but fc was not initialized.")
            flattened = self.flatten(embeddings)
            outputs = self.fc(flattened)

            if labels is not None:
                recon_loss = self.loss_function(outputs, labels)

        elif self.mode == "reconstruction":
            dec = F.interpolate(embeddings, size=(1, shape_before_pool2[3]), mode='nearest')

            dec = self.deconv_pointwise(dec)
            dec = self.deconv_seperable(dec)
            dec = self.batch_norm_dec_sep(dec)
            dec = self.activation(dec)

            dec = F.interpolate(dec, size=(1, shape_before_pool1[3]), mode='nearest')

            dec = self.deconv_depthwise(dec)
            dec = self.batch_norm_dec_depth(dec)
            dec = self.activation(dec)

            dec = self.deconv_temp(dec)
            outputs = dec.squeeze(1)  # (B, C, T)

            if labels is not None:
                recon_loss = F.mse_loss(outputs, labels)

        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'classification' or 'reconstruction'.")

        # Optional adversarial branch on embeddings
        if self.is_adversarial_training and self.adv_head is not None and adv_labels is not None:
            reversed_embeddings = GradReverse.apply(embeddings, self.adv_lambda)
            adv_logits = self.adv_head(reversed_embeddings)

            if adv_logits.shape[-1] == 1:
                adv_loss = F.binary_cross_entropy_with_logits(
                    adv_logits.squeeze(-1),
                    adv_labels.float()
                )
            else:
                adv_loss = F.cross_entropy(adv_logits, adv_labels.long())

        # Combine losses
        if recon_loss is not None and adv_loss is not None:
            loss = recon_loss + self.adv_lambda * adv_loss
        elif recon_loss is not None:
            loss = recon_loss
        elif adv_loss is not None:
            loss = self.adv_lambda * adv_loss

        return {
            "logits": outputs,
            "loss": loss,
            "recon_loss": recon_loss,
            "adv_loss": adv_loss,
            "adv_logits": adv_logits,
            "embeddings": embeddings if return_embeddings else None
        }

    def train(self, mode=True):
        super(EEGNetAutoEncoder, self).train(mode)

        if self.finetune_type == "frozen" and mode is True:
            super(EEGNetAutoEncoder, self).train(False)

            if self.fc is not None:
                self.fc.train(True)

            if self.adv_head is not None:
                self.adv_head.train(True)

        return self