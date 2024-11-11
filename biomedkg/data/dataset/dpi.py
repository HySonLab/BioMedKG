from typing import Callable

import pandas as pd

from ._base import TripletBase


class DPI(TripletBase):
    def __init__(self, data_dir: str, encoder: Callable = None):

        df = pd.read_csv(
            data_dir,
        )

        df = df.dropna()

        super().__init__(df=df, encoder=encoder)
