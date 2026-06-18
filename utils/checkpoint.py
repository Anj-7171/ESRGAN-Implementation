import torch, logging
log = logging.getLogger(__name__)


def save_checkpoint(G, opt_G, epoch, metric, path, D=None, opt_D=None):
    state = {"epoch": epoch, "metric": metric,
             "G_state": G.state_dict(), "opt_G": opt_G.state_dict()}
    if D:     state["D_state"] = D.state_dict()
    if opt_D: state["opt_D"]   = opt_D.state_dict()
    torch.save(state, path)
    log.info(f"Saved → {path}")


def load_checkpoint(path, G, opt_G=None, D=None, opt_D=None):
    ckpt = torch.load(path, map_location="cpu")
    G.load_state_dict(ckpt["G_state"])
    if opt_G and "opt_G" in ckpt: opt_G.load_state_dict(ckpt["opt_G"])
    if D     and "D_state" in ckpt: D.load_state_dict(ckpt["D_state"])
    if opt_D and "opt_D"  in ckpt: opt_D.load_state_dict(ckpt["opt_D"])
    return ckpt.get("epoch", 0), ckpt.get("metric", 0.0)