import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForMaskedLM

class Embedding(nn.Module):
    def __init__(self, tokeniser=None):
        super().__init__()
        if not tokeniser:
            self.tokeniser = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokeniser = tokeniser
        self.model = BertForMaskedLM.from_pretrained(
            '../models/bert-embed/bert', 
            output_hidden_states=True
        )
        self.model.eval()
    def forward(self, sentence: str):
        input_ids = torch.tensor([sentence['input_ids']])
        token_type_ids = torch.tensor([sentence['token_type_ids']])
        attention_mask = torch.tensor([sentence['attention_mask']])

        with torch.no_grad():
            output = self.model(input_ids, attention_mask, token_type_ids)
            hidden_states = output.hidden_states

            # Manipulate dims
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)

            """
            SENTENCE VEC
            """
            token_vecs = hidden_states[-2][0]
            # Calculate the average of all token vectors.
            sentence_embedding_output = torch.mean(token_vecs, dim=0)

            """
            TOKEN SUM VEC
            """
            token_vecs_sum = []
            # `token_embeddings` is a [len x 12 x 768] tensor.
            # For each token in the sentence...
            for token in token_embeddings:
                # `token` is a [12 x 768] tensor
                # Sum the vectors from the last four layers.
                sum_vec = torch.sum(token[-4:], dim=0)
                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(sum_vec)
            token_embedding_output = torch.vstack(token_vecs_sum)
            
            return {
                'Token': token_embedding_output,
                'Sentence': sentence_embedding_output
            }
