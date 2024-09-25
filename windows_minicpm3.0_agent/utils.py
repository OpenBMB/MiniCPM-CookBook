from torchvision import transforms
import torch.nn.functional as F
import torch
def evaluate_colbert( qs, ps, batch_size=128) -> torch.Tensor:
    scores = []
    for i in range(0, len(qs), batch_size):
        scores_batch = []
        qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
            "cuda"
        )
        qs_batch = F.normalize(qs_batch, p=2, dim=-1)
        for j in range(0, len(ps), batch_size):
            ps_batch = torch.nn.utils.rnn.pad_sequence(
                ps[j : j + batch_size], batch_first=True, padding_value=0
            ).to("cuda")
            ps_batch = F.normalize(ps_batch, p=2, dim=-1)
            scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores.append(scores_batch)
    scores = torch.cat(scores, dim=0)
    return scores
def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )