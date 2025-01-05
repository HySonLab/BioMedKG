# Biomedical Knowledge Graph

## Abstract

Biomedical Knowledge Graphs (BKGs) integrate diverse datasets to elucidate complex relationships within the biomedical field. Effective link prediction on these graphs can uncover valuable connections, such as potential novel drug-disease relations. We introduce a novel multimodal approach that unifies embeddings from specialized Language Models (LMs) with Graph Contrastive Learning (GCL) to enhance intra-entity relationships while employing a Knowledge Graph Embedding (KGE) model to capture inter-entity relationships for effective link prediction. To address limitations in existing BKGs, we present PrimeKG++, an enriched knowledge graph incorporating multimodal data, including biological sequences and textual descriptions for each entity type. By combining semantic and relational information in a unified representation, our approach demonstrates strong generalizability, enabling accurate link predictions even for unseen nodes. Experimental results on PrimeKG++ and the DrugBank drug-target interaction dataset demonstrate the effectiveness and robustness of our method across diverse biomedical datasets.

<p align="center">
   <img src="assets/framework_overview.png" width="75%"/><br/>
   <i>
    Overview of our proposed framework.
    <strong>A. Modality Embedding:</strong> Creating node attribute embeddings through domain-specific LMs.
    <strong>B. Contrastive Learning:</strong> Enhancement of LM-derived embeddings for specific node attributes of the same type through Fusion Module and Contrastive Learning.
    <strong>C. Link Prediction on KG Embedding:</strong> Utilizing the enhanced embeddings to perform link prediction tasks through a Knowledge Graph Embedding (KGE) model that learns relationships and enhances semantic information across distinct node types.
</i>

 </p>

## Setup

Install `huggingface-cli`:

```shell
pip install -U "huggingface_hub[cli]"
```

### Prepare data

Download **PrimeKG++** and **DrugBank DTI** datasets from the Hugging Face Hub, which include multimodal features and processed triplets:

```shell
huggingface-cli download tienda02/BioMedKG --repo-type=dataset --local-dir ./data
```

### Installation

Create a Conda environment and install dependencies:

```shell
conda create --name biokg python=3.10
conda activate biokg
make
```

## Usage

- Review configuration files in the [configs](./configs/) directory and training scripts in the [scripts](./scripts/) directory.
- Set your Comet API key to track experiments:

  `export COMET_API_KEY=<your-comet-api-key>`

- Modify the following variables for different as needed:
  - `NODE_TYPE`
  - `MODEL_NAME`
  - `NODE_INIT_METHOD`
  - `GCL_MODEL`
  - `GCL_FUSE_METHOD`
  - `PRETRAINED_PATH`

## Train

### PrimeKG++

#### Contrastive Learning

Train the model with graph contrastive learning to enhance intra-node type relationships:

```shell
bash scripts/gcl.sh
```

#### Link Prediction

Train the KGE model for link prediction tasks:

```shell
bash scripts/kge.sh
```

### DrugBank DTI

Fine-tune the KGE model on the Drug-Protein Interaction (DPI) benchmark:

```shell
bash scripts/dpi.sh
```

## Test

### PrimeKG++

Evaluate link prediction on the PrimeKG++ dataset:

```shell
bash scripts/test_kge.sh
```

### DrugBank DTI

Evaluate link prediction on the DrugBank DTI dataset:

```shell
bash scripts/test_dpi.sh
```

## Reproduce Results

To reproduce results, download the pre-trained checkpoints from the Hugging Face Hub:

```shell
huggingface-cli download tienda02/BioMedKG --repo-type=model --local-dir ./ckpt
```

Then, set the `PRETRAINED_PATH` variable in the test scripts to the downloaded checkpoint.
