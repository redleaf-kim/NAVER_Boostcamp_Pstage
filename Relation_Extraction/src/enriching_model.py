import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, XLMRobertaModel
from torch.nn import MSELoss, CrossEntropyLoss
from loss import LabelSmoothingCrossEntropy


def l2_loss(parameters):
    return torch.sum(
        torch.tensor([
            torch.sum(p ** 2) / 2 for p in parameters if p.requires_grad
        ]))


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, model_name=None):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = 42
        self.l2_reg_lambda = config.l2_reg_lambda
        if model_name is not None:
            self.bert = BertModel.from_pretrained(model_name, config=config)
        else:
            self.bert = BertModel(config)


        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_size = config.hidden_size*3
        self.classifier = nn.Linear(
            classifier_size, self.config.num_labels)
        self.latent_size = config.hidden_size
        self.latent_type = nn.Parameter(torch.FloatTensor(
            3, config.hidden_size), requires_grad=True)

        # self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None, cls_weight=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # for details, see the document of pytorch-transformer
        pooled_output = outputs[1]
        sequence_output = outputs[0]

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()
        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(weight=cls_weight)
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs




def l2_loss(parameters):
    return torch.sum(
        torch.tensor([
            torch.sum(p ** 2) / 2 for p in parameters if p.requires_grad
        ]))


class XLMRobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, model_name=None, smoothing=0.0):
        super(XLMRobertaForSequenceClassification, self).__init__(config)
        self.num_labels = 42
        self.l2_reg_lambda = config.l2_reg_lambda
        if model_name is not None:
            self.xlmroberta = XLMRobertaModel.from_pretrained(model_name, config=config)
        else:
            self.xlmroberta = XLMRobertaModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_size = config.hidden_size*3
        self.classifier = nn.Linear(
            classifier_size, self.config.num_labels)
        self.latent_size = config.hidden_size
        self.latent_type = nn.Parameter(torch.FloatTensor(
            3, config.hidden_size), requires_grad=True)

        if smoothing == 0.0: self.smoothing = False
        else: self.smoothing = True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, e1_mask=None, e2_mask=None, labels=None,
                position_ids=None, head_mask=None, cls_weight=None):
        outputs = self.xlmroberta(input_ids, position_ids=position_ids, attention_mask=attention_mask, head_mask=head_mask)
        # for details, see the document of pytorch-transformer
        pooled_output = outputs[1]
        sequence_output = outputs[0]

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output).squeeze(1)
            return extended_e_mask.float()
        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output)
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if not self.smoothing:
                    loss_fct = CrossEntropyLoss(weight=cls_weight)
                else:
                    loss_fct = LabelSmoothingCrossEntropy(weight=cls_weight)
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs