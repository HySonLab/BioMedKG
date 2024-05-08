# Biomedical Knowledge Graph

```
.
├── README.md
├── biomedkg
│   ├── configs
│   ├── modules
│   ├── data_module.py
│   └── kge_module.py 
├── data
│   ├── embed
│   │   └── *.pickle
│   ├── modalities
│   │   └── *.csv
│   └── primekg
├── modality.yaml
├── pyproject.toml
└── scripts
```
## Notes

* Configurations are initialized in [biomedkg/configs](biomedkg/configs), please modify/create for your needs. These variables are loaded from `.env` file, modify [.env-example](.env-example) to `.env` before any executions.

* Modules (such as model architecture, data module, node embedding, etc) should be developed inside [biomedkg/modules](biomedkg/modules).

* Lightning modules (Data module, Model module) are prepared in [biomedkg](biomedkg), check [data_module](biomedkg/data_module.py) and [kge_module](biomedkg/kge_module.py) for examples.

* All scripts should be created inside [scripts](scripts) folder.

## Setup

### Prepare data
Create a `data` folder, download enhanced modality csv file from Google Drive, and save it in `data/modalities`

### Installation
Run the followowing command to build the module and install dependencies.
```
pip install -e .
```

### Get Modality Embedding
This command extracts embeddings for each modality. Refer to [modality.yaml](./modality.yaml) for details, including model name and modality type, etc.
```
python scripts/get_modality_embedding.py
```

## Train
You should ```export COMET_API_KEY=<your-comet-api-key>``` to track the training process on Comet dash board, otherwise it will be recorded in `log` folder.

Run the following scripts with the flag `-h` for more information.
### Graph Contrastive Learning
To train a graph on contrastive learning with intra-node type, for example, with `gene`, try:
```
python scripts/train_gcl.py --model_name ggd --node_type gene
```
### Knowledge Graph Embedding
Run the following command to execute Knowledge Graph Embedding task
```
python scripts/train_kge.py
```

## Contributors:
* Viet Thanh Duy Nguyen
* Tuan Le
* Tien Dang
* Truong Son Hy (Correspondent / PI)
