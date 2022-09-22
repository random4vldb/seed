import os
from dataclasses import dataclass, field
from pyparsing import col

import torch
from nltk import wordpunct_tokenize
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import json
import inflect
import re
import datetime

inflect = inflect.engine()


def is_date(string):
    match = re.search('\d{4}-\d{2}-\d{2}', string)
    if match:
        try:
            date = datetime.datetime.strptime(match.group(), '%Y-%m-%d').date()
        except:
            return False
        return True
    else:
        return False


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, labels):
        """Constructor
        Input: in_dim	- Dimension of input vector
                   out_dim	- Dimension of output vector
                   vocab	- Vocabulary of the embedding
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.drop = torch.nn.Dropout(0.2)
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

class InfotabVerifier:
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained("roberta-base").cuda()
        embed_size = self.model.config.hidden_size
        self.classifier = FeedForward(embed_size, int(embed_size / 2), 3).cuda()

        # Load pre-trained models
        checkpoint = torch.load("models/infotab/model_3_0.9092581238503985")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")


    def json_to_para(self, data):
        result = []

        for index, row in enumerate(data):
            obj = json.loads(row["table"])
            if "index" in obj:
                obj.drop("index")
            print("Obj: ", obj)

            if not obj:
                continue
            obj = obj[0]
            obj = {x:y if isinstance(y, list) else [y] for x,y in obj.items()}

            try:
                title = row["title"]
            except KeyError as e:
                print(row)
                exit()


            para = ""

            if "index" in obj:
                obj.pop("index")

            for key in obj:
                line = ""
                values = obj[key]


                if (len(values) > 1) and (inflect.plural_noun(key)):
                    verb_use = "are"
                    if is_date("".join(values)):
                        para += title + " was " + str(key) + " on "
                        line += title + " was " + str(key) + " on "
                    else:
                        try:
                            para += (
                                "The " + str(key) + " of " + title + " " + verb_use + " "
                            )
                            line += (
                                "The " + str(key) + " of " + title + " " + verb_use + " "
                            )
                        except TypeError as e:
                            print(e)
                            print(row)
                            print(key)
                            print(title)
                            exit()
                    for value in values[:-1]:
                        para += value + ", "
                        line += value + ", "
                    if len(values) > 1:
                        para += "and " + values[-1] + ". "
                        line += "and " + values[-1] + ". "
                    else:
                        para += values[-1] + ". "
                        line += values[-1] + ". "
                else:
                    verb_use = "is"
                    if is_date(values[0]):
                        para += title + " was " + str(key) + " on " + values[0] + ". "
                        line += title + " was " + str(key) + " on " + values[0] + ". "
                    else:
                        para += (
                            "The "
                            + str(key)
                            + " of "
                            + title
                            + " "
                            + verb_use
                            + " "
                            + values[0]
                            + ". "
                        )
                        line += (
                            "The "
                            + str(key)
                            + " of "
                            + title
                            + " "
                            + verb_use
                            + " "
                            + values[0]
                            + ". "
                        )

            label = row["label"]

            obj = {
                "index": index,
                "table_id": row["table_id"],
                "annotator_id": row["annotator_id"],
                "premise": para,
                "hypothesis": row["hypothesis"],
                "label": label,
            }
            print(obj)
            result.append(obj)
        return result


    def preprocess(self, data):
        keys = ["uid", "encodings", "attention_mask", "segments", "labels"]
        data_dict = {key: [] for key in keys}
        samples_processed = 0
        # Iterate over all data points
        for pt_dict in data:

            samples_processed += 1
            # Encode data. The premise and hypothesis are encoded as two different segments. The
            # maximum length is chosen as 504, i.e, 500 sub-word tokens and 4 special characters
            # If there are more than 504 sub-word tokens, sub-word tokens will be dropped from
            # the end of the longest sequence in the two (most likely the premise)

            encoded_inps = self.tokenizer(
                pt_dict["premise"],
                pt_dict["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=504,
            )

            # Some models do not return token_type_ids and hence
            # we just return a list of zeros for them. This is just
            # required for completeness.
            if "token_type_ids" not in encoded_inps.keys():
                encoded_inps["token_type_ids"] = [0] * len(encoded_inps["input_ids"])

            data_dict["uid"].append(int(pt_dict["index"]))
            data_dict["encodings"].append(encoded_inps["input_ids"])
            data_dict["attention_mask"].append(encoded_inps["attention_mask"])
            data_dict["segments"].append(encoded_inps["token_type_ids"])
            data_dict["labels"].append(pt_dict["label"])

        return data_dict


    def encode(self, sentence, df):
        data = [{
                "table_id": 0,
                "annotator_id": 0,
                "hypothesis": sentence,
                "table": df.to_json(orient="records"),
                "label": 0,
                "title": df.loc[0, df.columns[0]],
            }]
        result = self.json_to_para(data)
        return self.preprocess(result)


        
    def text_similarity(self, doc, table):
        list1 = wordpunct_tokenize(doc)
        list2 = wordpunct_tokenize(" ".join(table.astype(str).values.tolist()[0]))
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def verify(self, sentence, df):
        if self.text_similarity(sentence, df) < 0.01:
            return None

        column2res = {}


        data =self.encode(sentence, df)

        enc = torch.tensor(data["encodings"]).cuda()
        mask = torch.tensor(data["attention_mask"]).cuda()
        segs = torch.tensor(data["segments"]).cuda()
        labs = torch.tensor(data["labels"]).cuda()
        ids = torch.tensor(data["uid"]).cuda()

        # Create Data Loader for the split

        self.model.eval()

            # Forward-pass w/o calculating gradients
        with torch.no_grad():
            outputs = self.model(enc, attention_mask=mask, token_type_ids=segs)
            predictions = self.classifier(outputs[1])

            # Calculate metrics
            _, inds = torch.max(predictions, 1)
            probs = torch.softmax(predictions, 1)
            predicted_class_idx = inds


            if predicted_class_idx != 1:
                column2res["Test"] = (predicted_class_idx, torch.max(probs).item())
            if predicted_class_idx != 1:
                column2res["Test"] = (predicted_class_idx, torch.max(probs).item())
        return column2res