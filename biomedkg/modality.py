from biomedkg.configs import node_settings
from biomedkg.modules.fusion import AttentionFusion, ReDAF

class ModalityFuserFactory:
    @staticmethod
    def create_fuser(method: str):
        if method == "attention":
            return AttentionFusion(
                    embed_dim=node_settings.PRETRAINED_NODE_DIM,
                    norm=True,
                )
        elif method == "redaf":
            return ReDAF(
                embed_dim=node_settings.PRETRAINED_NODE_DIM,
                num_modalities = 2,
            )     
        else:
            return None