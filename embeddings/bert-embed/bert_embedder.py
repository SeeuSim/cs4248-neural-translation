import torch
from bert_tokeniser import BPEBertTokeniser
from transformers import BertForMaskedLM

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BPEBertEmbedder:
    def __init__(self, lang: str, model_file: str, spm_file: str):
        # Init the model from the pretrained weights
        self.model = BertForMaskedLM.from_pretrained(
            model_file, output_hidden_states=True
        ).to(DEVICE)
        self.model.eval()

        # Init the BPE tokeniser (padded to length 288, vocab 16384)
        self.tokeniser = BPEBertTokeniser(lang, model_file=spm_file)
        pass

    def embed(self, sent: str):
        tokens = self.tokeniser(
            sent
        )  # input_ids, token_type_mask, attention_mask, special_tokens_mask
        input_ids = torch.tensor([tokens["input_ids"]]).to(DEVICE)
        token_type_ids = torch.tensor([tokens["token_type_ids"]]).to(DEVICE)
        attention_mask = torch.tensor([tokens["attention_mask"]]).to(DEVICE)

        with torch.no_grad():
            # Perform inference
            output = self.model(input_ids, token_type_ids, attention_mask)
            hidden_states = output.hidden_states

            # Permutate and obtain hidden states
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1, 0, 2)  # (288, n_layers, 256)

            # Take last 4 layers
            token_embeddings = token_embeddings[:, -4:, :]

            # Take sum of last 4 layers
            token_embeddings = token_embeddings.sum(axis=1)  # (288, 256)

            # Matrix of embeddings of dim 256, one per word
            return token_embeddings
