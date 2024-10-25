from typing import Callable

import pandas as pd

from ._base import TripletBase


class BioKG(TripletBase):
    def __init__(self, data_dir: str, encoder: Callable = None):

        df = pd.read_csv(
            data_dir,
        )

        super().__init__(df=df, encoder=encoder)
