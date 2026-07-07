from ms_uq.loss import FPLoss
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, CosineSimilarity
from torch_geometric.utils import unbatch
import massspecgym.utils as utils
from torchmetrics.functional.retrieval import retrieval_hit_rate


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs=990,
        n_outputs=256,
        layer_dims=[1024, 512],
        layer_or_batchnorm="layer",
        dropout=0.25,
    ):
        super().__init__()

        c = n_inputs
        layers = []
        for i in layer_dims:
            layers.append(nn.Linear(c, i))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(
                nn.LayerNorm(i) if layer_or_batchnorm == "layer" else nn.BatchNorm1d(i)
            )
            c = i

        layers.append(nn.Linear(c, n_outputs))

        self.net = nn.Sequential(*layers)

        self.hsize = n_outputs

    def forward(self, x):
        return self.net(x)


class FingerprintPredicter(RetrievalMassSpecGymModel):
    def __init__(
        self,
        n_in=1000,  # number of bins
        layer_dims=[512],  # hidden layer sizes
        n_bits=4096,  # fingerprint size
        layer_or_batchnorm="layer",
        dropout=0.2,
        bitwise_loss=None,  # "bce", "fl"
        fpwise_loss=None,  # "cossim", "iou"
        rankwise_loss=None,  # "bienc", "cross"
        bitwise_lambd=1.0,
        fpwise_lambd=1.0,
        rankwise_lambd=1.0,
        bitwise_kwargs={},  # {"weighted" : False} / {"weighted" : False, "gamma" : 2}
        fpwise_kwargs={},  # {} / {"jml_version" : True}
        rankwise_kwargs={},  # {"temp": 0.1, "n_bits" : 4096, "dropout":0.2, sim_func:"cossim", "projector":False} / all same without sim_func
        mc_dropout_eval=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mc_dropout_eval = mc_dropout_eval
        self.mlp = MLP(
            n_inputs=n_in,
            n_outputs=layer_dims[-1],
            layer_dims=layer_dims[:-1],
            layer_or_batchnorm=layer_or_batchnorm,
            dropout=dropout,
        )
        self.loss = FPLoss(
            layer_dims[-1],
            n_bits,
            bitwise_loss=bitwise_loss,
            fpwise_loss=fpwise_loss,
            rankwise_loss=rankwise_loss,
            bitwise_lambd=bitwise_lambd,
            fpwise_lambd=fpwise_lambd,
            rankwise_lambd=rankwise_lambd,
            bitwise_kwargs=bitwise_kwargs,
            fpwise_kwargs=fpwise_kwargs,
            rankwise_kwargs=rankwise_kwargs,
        )

    # function to enable mc dropout in evaluation
    def _enable_mc_dropout(self):
        if self.mc_dropout_eval and not self.training:  # i.e. eval() mode
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()  # activate masks

    def encode_spectrum(self, x):
        return self.mlp(x)

    def forward(self, x):
        # check if mc dropout should be enabled
        self._enable_mc_dropout()
        return self.encode_spectrum(x)

    def step(self, batch, stage):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = batch["spec"]
        fp_true = batch["mol"]
        cands = batch["candidates"].int()
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        embedding = self(x)

        # Calculate loss
        loss = self.loss(embedding, fp_true, cands, batch_ptr, batch["labels"])

        return dict(loss=loss)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "train_loss",
            outputs["loss"],
            batch_size=batch["spec"].size(0),
            sync_dist=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        x = batch["spec"]
        fp_true = batch["mol"]
        cands = batch["candidates"].int()
        batch_ptr = batch["batch_ptr"]

        # Predict fingerprint
        embedding = self(x)

        # Calculate loss
        loss = self.loss(embedding, fp_true, cands, batch_ptr, batch["labels"])

        fp_pred = F.sigmoid(self.loss.fp_pred_head(embedding))
        # Evaluation performance on fingerprint prediction (optional)

        tanimotos = batch_samplewise_tanimoto(fp_pred, fp_true, reduce=False)
        cont_ious = cont_iou(fp_pred, fp_true)

        self._update_metric(
            "val_fingerprint_av_contiou",
            MeanMetric,
            (cont_ious,),
            batch_size=fp_true.size(0),
        )

        self._update_metric(
            "val_fingerprint_av_tanim",
            MeanMetric,
            (tanimotos,),
            batch_size=fp_true.size(0),
        )
        self._update_metric(
            "val_fingerprint_perc_close_match",
            MeanMetric,
            ((tanimotos > 0.675).to(float),),
            batch_size=fp_true.size(0),
        )
        self._update_metric(
            "val_fingerprint_perc_meaningful_match",
            MeanMetric,
            ((tanimotos > 0.40).to(float),),
            batch_size=fp_true.size(0),
        )

        self._update_metric(
            "val_fingerprint_av_cossim",
            CosineSimilarity,
            (fp_pred, fp_true),
            batch_size=fp_true.size(0),
            metric_kwargs=dict(reduction="mean"),
        )

        # Calculate final similarity scores between predicted fingerprints and corresponding
        # candidate fingerprints for retrieval
        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)

        scores = nn.functional.cosine_similarity(fp_pred_repeated, cands)
        self.evaluate_retrieval_step(
            scores,
            batch["labels"],
            batch["batch_ptr"],
            stage=Stage("val"),
            name="cossim",
        )

        scores = batch_samplewise_tanimoto(fp_pred_repeated, cands)
        self.evaluate_retrieval_step(
            scores,
            batch["labels"],
            batch["batch_ptr"],
            stage=Stage("val"),
            name="tanim",
        )

        scores = cont_iou(fp_pred_repeated, cands)
        self.evaluate_retrieval_step(
            scores,
            batch["labels"],
            batch["batch_ptr"],
            stage=Stage("val"),
            name="contiou",
        )

        if self.loss.rankwise_loss:
            scores = self.loss.ranker(fp_pred_repeated, cands)
            self.evaluate_retrieval_step(
                scores,
                batch["labels"],
                batch["batch_ptr"],
                stage=Stage("val"),
                name="reranker",
            )

        return dict(loss=loss)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "val_loss",
            outputs["loss"],
            batch_size=batch["spec"].size(0),
            sync_dist=True,
            prog_bar=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mc_dropout_eval:
            # Keep dropout layers active during evaluation
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        return self

    def test_step(self, batch, batch_idx):
        # Forward to the already‑implemented validation logic
        return self.validation_step(batch, batch_idx)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        # Re‑use the logging hook, too
        self.on_validation_batch_end(outputs, batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["spec"]
        emb = self(x)
        return torch.sigmoid(self.loss.fp_pred_head(emb))  # (B,4096)

    def evaluate_retrieval_step(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        batch_ptr: torch.Tensor,
        stage: Stage,
        name: str,
    ) -> dict[str, torch.Tensor]:
        # Initialize return dictionary to store metric values per sample
        metric_vals = {}

        # This makes it so that - in the event all scores are equal - not always the first element is sorted as the top prediction
        some_noise = torch.randn_like(scores) * torch.finfo(scores.dtype).eps
        scores_w_noise = scores + some_noise

        # Evaluate hitrate at different top-k values
        indexes = utils.batch_ptr_to_batch_idx(batch_ptr)
        scores = unbatch(scores_w_noise, indexes)
        labels = unbatch(labels, indexes)

        for at_k in self.at_ks:
            hit_rates = []
            for scores_sample, labels_sample in zip(scores, labels):
                hit_rates.append(
                    retrieval_hit_rate(scores_sample, labels_sample, top_k=at_k)
                )
            hit_rates = torch.tensor(hit_rates, device=batch_ptr.device)

            metric_name = f"{stage.to_pref()}{name}_hit_rate@{at_k}"
            self._update_metric(
                metric_name,
                MeanMetric,
                (hit_rates,),
                batch_size=batch_ptr.size(0),
                bootstrap=stage == Stage.TEST,
            )
            metric_vals[metric_name] = hit_rates

        return metric_vals


def batch_samplewise_tanimoto(pred_fp, true_fp, threshold=0.5, reduce=False):
    _and = (true_fp.int() & (pred_fp > threshold)).sum(-1)
    _or = (true_fp.int() | (pred_fp > threshold)).sum(-1)
    if reduce:
        return (_and / _or).mean()
    else:
        return _and / _or


def cont_iou(fp_pred, fp_true):
    total = (fp_pred + fp_true.to(fp_pred.dtype)).sum(-1)
    difference = (fp_pred - fp_true.to(fp_pred.dtype)).abs().sum(-1)
    return (total - difference) / (total + difference)
