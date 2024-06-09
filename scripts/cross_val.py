
# Load dataframe: input dir ("dpi-fda.csv")-> df 
# Split using sklearn -> cross validation aka k-fold: output 2 dfs

from sklearn.model_selection import KFold
import pandas as pd 
from biomedkg.modules.data import TripletBase

def main(data_dir: str):
    df = pd.read_csv(data_dir)
    kfold = KFold(n_splits=5) # k 
    kfold.get_n_splits(df)

    for i, (train_index, test_index) in enumerate(kfold.split(df)):
        train_data, test_data = df.iloc[train_index], df.iloc[test_index]
        # Run the rest of the code here 

        print(f"Fold {i + 1}:")

        train_graph, test_graph = TripletBase(df=train_data, embed_dim = 768), TripletBase(df=test_data, embed_dim = 768)

        print(train_graph, test_graph)

main("data\dpi_fda.csv")