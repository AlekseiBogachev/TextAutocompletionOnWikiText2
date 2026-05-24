import torch


class Perplexity:
    def __init__(self, **cross_entropy_kwargs):
        self.cross_entropy = torch.nn.CrossEntropyLoss(**cross_entropy_kwargs)

    def __call__(self, logits, labels):
        return torch.exp(self.cross_entropy(logits, labels))


class Accuracy:
    def __init__(self, batch_agg="sum"):
        self.batch_agg = batch_agg

    def __call__(self, logits, labels):
        preds = torch.argmax(logits, dim=-1)
        if self.batch_agg == "sum":
            return (preds == labels).sum()
        elif self.batch_agg == "mean":
            return (preds == labels).mean()
        else:
            return (preds == labels)
