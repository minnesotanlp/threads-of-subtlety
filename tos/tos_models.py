from typing import Optional

import torch
from transformers import LongformerModel
from transformers.modeling_outputs import SequenceClassifierOutput


class LongformerWithMotifsClassificationHead(torch.nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, num_labels, motif_dims):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size + motif_dims, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, motif_dist):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states_with_motifs = torch.cat(
            (hidden_states, motif_dist), dim=-1
        )  # [batch size, hidden size+num extra dims]
        hidden_states_with_motifs = self.dropout(hidden_states_with_motifs)
        hidden_states_with_motifs = self.dense(hidden_states_with_motifs)
        hidden_states_with_motifs = torch.tanh(hidden_states_with_motifs)
        hidden_states_with_motifs = self.dropout(hidden_states_with_motifs)
        output = self.out_proj(hidden_states_with_motifs)
        return output


class LongformerWithMotifsForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name="allenai/longformer-base-4096", num_labels=2):
        super().__init__()
        # self.motif_dims = 31 + 96 + 80
        self.motif_dims = (31 + 96 + 80) * 2
        self.num_labels = num_labels

        self.longformer = LongformerModel.from_pretrained(
            model_name, num_labels=self.num_labels, add_pooling_layer=False
        )
        self.hidden_size = self.longformer.config.hidden_size
        self.hidden_dropout_prob = self.longformer.config.hidden_dropout_prob
        self.classifier = LongformerWithMotifsClassificationHead(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            num_labels=self.num_labels,
            motif_dims=self.motif_dims,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        motif_dist: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=True,
        )
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output, motif_dist)

        loss = None
        if labels is not None:
            labels = labels.reshape(-1, 1)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
