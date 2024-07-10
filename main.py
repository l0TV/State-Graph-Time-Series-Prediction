from model.state_graph import StateGraph
from dataloader import Loader
from utils import *
from trainer import Experiment

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = Loader()
    model = StateGraph(device).to(device)

    exp = Experiment(model, data_loader, device)
    if cfg.train:
        exp.train()
        exp.test()
        exp.test(is_last_model=True)
    else:
        exp.test()

