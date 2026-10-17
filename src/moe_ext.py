# moe_ext.py
#
# Une las dos ampliaciones de la práctica sin modificar ninguna:
#   - codificaciones posicionales intercambiables (model_ext.GPTModelExt, del equipo)
#   - Mixture of Experts en el feed-forward (moe.py)
#
# La codificación posicional actúa en la atención y en la entrada; el MoE sustituye el
# FeedForward de cada bloque (block.ff). Son módulos distintos del bloque transformer,
# así que se combinan con attach_moe sobre un GPTModelExt ya construido.
#
# build_ext_model(cfg) sirve de model_factory para model_base_adaptado.pretrain y
# load_checkpoint: si cfg no tiene "moe" devuelve el GPTModelExt del equipo tal cual.
# Con "moe", cada capa MoE inyecta la pérdida de balanceo en el backward
# (moe._AuxLossInjector), porque pretrain solo optimiza la entropía cruzada.
#
#   cfg = make_config(get_small_config(65), "alibi", moe={"n_experts": 4, "top_k": 1,
#                                                         "aux_loss_coef": 0.01})
#   res = pretrain(cfg, text, tok, model_factory=build_ext_model, seed=0)

from model_ext import GPTModelExt
from moe import attach_moe


def build_ext_model(cfg):
    """GPTModelExt con la codificación cfg["pos_encoding"] y, si cfg tiene "moe",
    capas MoE en el feed-forward que inyectan su pérdida de balanceo."""
    model = GPTModelExt(cfg)
    if "moe" in cfg:
        attach_moe(model, cfg, inject_aux_loss=True)
    return model
