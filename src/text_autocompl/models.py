from copy import deepcopy

import torch


class RecurWithRes(torch.nn.Module):
    def __init__(self, cell_type="LSTM", **cell_kwargs):
        super().__init__()
        cell_params = deepcopy(cell_kwargs)
        cell_params["batch_first"] = True
        cell_params["num_layers"] = 1
        dropout_p = cell_params.pop("dropout", 0.0)

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
        output = self.dropout(output) + X

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


        self.out_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
        )

    def forward(self, input_ids):
        res = self.embedding(input_ids)
        res = self.input_proj(res)

        for layer in self.rec_layers:
            res, _ = layer(res)
        
        res = self.out_linear(res)

        return res
