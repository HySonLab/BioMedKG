import torch
from typing import List
from transformers import AutoTokenizer, AutoModel, BertConfig
from transformers import AutoTokenizer, AutoModel

class NodeEmbedding:
    def __init__(self, model_name_or_path : str):
        
        self.device = self.find_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if "DNA" in model_name_or_path:
            config = BertConfig.from_pretrained(
                model_name_or_path, 
                local_files_only=True
                )
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                device_map=self.device, 
                config=config
                )
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                device_map=self.device, 
                trust_remote_code=True
                )

    def __call__(self, input_lst : List[str]) -> torch.tensor:
        input_ids = self.tokenizer(
            input_lst, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
            ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(**input_ids)

        if isinstance(embeddings, tuple):
            hidden_state = embeddings[0]
        else:
            hidden_state = embeddings.last_hidden_state[:,0,:]
        
        return hidden_state.detach().cpu()

    @staticmethod
    def find_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    