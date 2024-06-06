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
Create a `data` folder, download an enhanced modality CSV file from Google Drive, and save it in `data/modalities`

### Installation
Run the following command to build the module and install dependencies.
```
pip install -e .
```

### Get Modality Embedding
This command extracts embeddings for each modality. Refer to [modality.yaml](./modality.yaml) for details, including model name and modality type, etc.
```
python scripts/get_modality_embedding.py
```

## Train
You should ```export COMET_API_KEY=<your-comet-api-key>``` to track the training process on Comet dashboard, otherwise, it will be recorded in `log` folder.

Run the following scripts with the flag `-h` for more information.
### Graph Contrastive Learning
To train a graph on contrastive learning with intra-node type, for example, with `gene`, try:
```
python scripts/train_gcl.py --model_name ggd --node_type gene
```

### Get node embedding after training GCL
To obtain the node embedding for each modality after training with GCL, execute the following command to load the checkpoint and extract the embedding:
```
python scripts/get_embedding_from_ckpt.py --model_name ggd --ckpt ./ckpt/gcl/*/*.ckpt
```

### Knowledge Graph Embedding
Run the following command to execute Link Prediction on Knowledge Graph Embedding:
```
python scripts/train_kge.py
```

## Benchmark on FDI-DPA

### Additional data requirements:
* Download `dpi_fda.csv` to `data/benchmarks/`
* Download `dpi_drug_feature.csv` to `data/modalities/`
* Download `dpi_gene_feature.csv` to `data/modalities/`


### Benchmark 
Run the following command to execute Link Prediction on DPI-FDA dataset:
```
python scripts/train_kge.py --gcl_embed_path ./data/gcl_embed/*/ --ckpt_path ./ckpt/kge/*/*.ckpt --run_benchmark
```

## Contributors:
* Viet Thanh Duy Nguyen
* Tuan Le
* Tien Dang
* Truong Son Hy (Correspondent / PI)
