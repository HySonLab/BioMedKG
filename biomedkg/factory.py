from omegaconf import DictConfig
from torch_geometric.nn import GAE

from .model import RGAT, RGCN, ComplEx, DistMult, TransE
from .utils.fusion import AttentionFusion, ReDAF


class FusionFactory:
    @staticmethod
    def create_fuser(method: str, embed_dim):
        if method == "attention":
            return AttentionFusion(embed_dim=embed_dim)
        if method == "redaf":
            return ReDAF(embed_dim=embed_dim)
        return None


class KGEModelFactory:
    @staticmethod
    def get_model(
        encoder_name: str,
        decoder_name: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        num_relation: int,
        num_heads: int = None,
    ):

        encoder = KGEModelFactory._get_encoder(
            encoder_name=encoder_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_hidden_layers=num_hidden_layers,
            num_relation=num_relation,
            num_heads=num_heads,
        )

        decoder = KGEModelFactory._get_decoder(
            decoder_name=decoder_name,
            num_relation=num_relation,
            hidden_channels=out_dim,
        )

        return GAE(
            encoder=encoder,
            decoder=decoder,
        )

    @staticmethod
    def _get_encoder(
        encoder_name: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_layers: int,
        num_relation: int,
        num_heads: int = None,
    ):
        if encoder_name == "rgcn":
            return RGCN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_relations=num_relation,
            )

        if encoder_name == "rgat":
            return RGAT(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_hidden_layers=num_hidden_layers,
                num_heads=num_heads,
                num_relations=num_relation,
            )

    @staticmethod
    def _get_decoder(
        decoder_name: str,
        num_relation: int,
        hidden_channels: int,
    ):
        if decoder_name == "transe":
            return TransE(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )
        if decoder_name == "dismult":
            return DistMult(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )
        if decoder_name == "complex":
            return ComplEx(
                num_relations=num_relation,
                hidden_channels=hidden_channels,
            )


def create_kge_model(cfg: DictConfig):
    return KGEModelFactory.get_model(
        encoder_name=cfg.encoder_name,
        decoder_name=cfg.decoder_name,
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.out_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_relation=cfg.num_relation,
        num_heads=cfg.num_heads,
    )
