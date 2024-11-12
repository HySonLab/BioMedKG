import os
from typing import Callable

import pandas as pd

from ._base import TripletBase


class PrimeKG(TripletBase):
    def __init__(
        self,
        data_dir: str,
        node_type: list[str],
        encoder: Callable = None,
    ):

        try:
            from tdc.resource import PrimeKG

            primekg = PrimeKG(path=data_dir)
            df = primekg.df

        except ModuleNotFoundError:
            csv_path = f"{data_dir}/kg.csv"

            if not os.path.exists(csv_path):
                os.system(
                    f"wget -O {csv_path} https://dataverse.harvard.edu/api/access/datafile/6180620"
                )

            df = pd.read_csv(csv_path, low_memory=False)

        if node_type:
            df = df[
                df["x_type"].isin(list(node_type)) & df["y_type"].isin(list(node_type))
            ]

        super().__init__(df=df, encoder=encoder)
