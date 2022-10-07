import torch.nn as nn
import torch.nn.functional as F
from tango.integrations.torch import Model
from transformers import AutoConfig, AutoModel
import torch

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, labels):
        """Constructor
        Input: in_dim	- Dimension of input vector
                   out_dim	- Dimension of output vector
                   vocab	- Vocabulary of the embedding
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(out_dim, labels)
        # self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        """Function for forward pass
        Input:	inp 	- Input to the network of dimension in_dim
        Output: output 	- Output of the network with dimension vocab
        """
        out_intermediate = F.relu(self.fc1(inp))
        output = self.fc2(out_intermediate)
        return output


@Model.register("infotab")
class InfotabModel(Model):
    def __init__(self, model_name_or_path) -> None:
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        embed_size = self.model.config.hidden_size
        self.classifier = FeedForward(embed_size, int(embed_size / 2), 2).cuda()

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        loss_fn = nn.CrossEntropyLoss()

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        outputs = self.classifier(outputs[1])
        loss = loss_fn(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        return {"loss": loss, "predictions": predictions, "labels": labels}
    


