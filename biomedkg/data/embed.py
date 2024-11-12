from typing import List

import torch
from transformers import AutoModel, AutoTokenizer, BertConfig

from biomedkg.common import find_device


class NodeEmbedding:
    def __init__(self, model_name_or_path: str):

        self.device = find_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        if "DNA" in model_name_or_path:
            config = BertConfig.from_pretrained(
                model_name_or_path, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name_or_path, device_map=self.device, config=config
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                device_map=self.device,
                trust_remote_code=True,
            )

    def __call__(self, input_lst: List[str]) -> torch.Tensor:
        input_ids = self.tokenizer(
            input_lst,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model(**input_ids)

        if isinstance(embeddings, tuple):
            hidden_state = embeddings[0]
        else:
            hidden_state = embeddings.last_hidden_state[:, 0, :]

        return hidden_state.detach().cpu()
