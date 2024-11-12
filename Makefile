build:
	pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
	pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
	pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
	pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
	pip install -e .
