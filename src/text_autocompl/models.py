from copy import deepcopy

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurWithRes(torch.nn.Module):
    def __init__(self, cell_type="LSTM", **cell_kwargs):
        super().__init__()
        cell_params = deepcopy(cell_kwargs)
        cell_params["batch_first"] = True
        cell_params["num_layers"] = 1
        dropout_p = cell_params.pop("dropout", 0.0)

        self.norm_layer = torch.nn.LayerNorm(cell_kwargs["hidden_size"])
        self.dropout = torch.nn.Dropout(p=dropout_p)

        if cell_type == "LSTM":
            cell_cls = torch.nn.LSTM
        elif cell_type == "GRU":
            cell_cls = torch.nn.GRU
        else:
            raise ValueError("Support only LSTM and GRU")

        self.recurrent_cell = cell_cls(**cell_params)

    def forward(self, X, hidden_state=None):
        output, hidden_state = self.recurrent_cell(X, hidden_state)

        # Чтобы не применять LayerNorm и Dropout к падингу и не делать
        # распаковку из упаковку последовательности повторно, обратимся
        # к данным напрямую
        out_data = self.norm_layer(output.data)
        out_data = self.dropout(out_data) + X.data
        output = torch.nn.utils.rnn.PackedSequence(
            out_data,
            output.batch_sizes,
            output.sorted_indices,
            output.unsorted_indices,
        )

        return output, hidden_state


class RecNN(torch.nn.Module):
    def __init__(
        self,
        cell_type="LSTM",
        embedding_size=128,
        vocab_size=10000,
        pad_idx=0,
        n_rec_layers=2,
        hidden_size=256,
        dropout=0,
        resid_connections=False,
    ):
        super().__init__()

        if cell_type == "LSTM":
            cell_cls = torch.nn.LSTM
        elif cell_type == "GRU":
            cell_cls = torch.nn.GRU
        else:
            raise ValueError("Support only LSTM and GRU")

        self.pad_idx = pad_idx

        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            padding_idx=pad_idx,
        )

        if resid_connections and embedding_size != hidden_size:
            self.input_proj = torch.nn.Linear(embedding_size, hidden_size)
        else:
            self.input_proj = torch.nn.Identity()

        self.rec_layers = torch.nn.ModuleList()

        if resid_connections:
            for _ in range(n_rec_layers):
                self.rec_layers.append(
                    RecurWithRes(
                        cell_type=cell_type,
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        dropout=dropout,
                    )
                )

            self.norm_layer = None
        else:
            self.rec_layers.append(
                cell_cls(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=n_rec_layers,
                    batch_first=True,
                    dropout=dropout,
                )
            )

            self.norm_layer = torch.nn.LayerNorm(hidden_size)

        self.out_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

    def forward(self, input_ids, lengths):
        res = self.embedding(input_ids)
        res = self.input_proj(res)

        res = pack_padded_sequence(
            res,
            lengths,
            batch_first=True,
            enforce_sorted=True,
        )

        for layer in self.rec_layers:
            res, _ = layer(res)

        if self.norm_layer is not None:
            normed_res = self.norm_layer(res.data)
            res = torch.nn.utils.rnn.PackedSequence(
                data=normed_res,
                batch_sizes=res.batch_sizes,
                sorted_indices=res.sorted_indices,
                unsorted_indices=res.unsorted_indices,
            )

        res, _ = pad_packed_sequence(
            res, batch_first=True, padding_value=self.pad_idx
        )

        res = self.out_linear(res)

        return res
