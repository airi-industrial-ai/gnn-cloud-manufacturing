import torch
import wandb
import numpy as np
from torch.optim import Adam
from tqdm.auto import trange
import torch.nn.functional as F
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split


from cloudmanufacturing.graphconv import GNN
from cloudmanufacturing.graph import os_type, so_type, ss_type
from exp_validation import validate_objective
from data_preparation import GraphDataset

class Trainer():
    def __init__(self, wandb_init_params) -> None:
        self.tracker = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run = wandb.init(**wandb_init_params, config={})

    def create_graph_dataset(self, info_file, train_size=0.8):
        train_idx, test_idx = train_test_split(
            list(info_file.keys()),
            random_state=42,train_size=train_size
        )
        print('train size: ', len(train_idx))
        print('test size: ', len(test_idx))
        self.train_dataset = GraphDataset([info_file[i]['problem'] for i in train_idx],
                                [info_file[i]['gamma'] for i in train_idx],
                                [info_file[i]['delta'] for i in train_idx])

        self.test_dataset = GraphDataset([info_file[i]['problem'] for i in test_idx],
                                    [info_file[i]['gamma'] for i in test_idx],
                                    [info_file[i]['delta'] for i in test_idx])
        
        wandb.config.update({'train_size':train_size,
                            'train_samples':len(train_idx),
                            'test_samples':len(test_idx)})

    def create_dataloader(self, info_file, train_size=0.8, 
                          train_batch=100, test_batch=20):
        self.create_graph_dataset(info_file, train_size=train_size)
        self.train_loader = GraphDataLoader(
            self.train_dataset, batch_size=train_batch, shuffle=True
        )
        self.test_loader = GraphDataLoader(
            self.test_dataset, batch_size=test_batch, shuffle=True
        )
        wandb.config.update({'train_batch':train_batch,
                            'test_batch':test_batch})

    def create_model(self, out_dim, lr, layers,
                     sheduler=False, scheduler_gamma=0.98):
        model = GNN(s_shape_init=1, o_shape_init=20, os_shape_init=2,
                ss_shape_init=10, out_dim=out_dim, n_layers=layers).to(self.device)
        optim = Adam(model.parameters(), lr=lr)
        if sheduler:
            scheduler = ExponentialLR(optim, gamma=scheduler_gamma)
            self.model = model
            self.optim = optim
            self.scheduler = scheduler
        else:
            self.model = model
            self.optim = optim

        wandb.config.update({
            'out_dim':out_dim,
            'layers':layers,
            'lr':lr,
            'sheduler':sheduler,
            'scheduler_gamma':scheduler_gamma
        })

    def train_epoch(self, service_rate):
        train_loss = []
        for graph in self.train_loader:
            self.optim.zero_grad()
            gamma_target = graph.edata["target"][os_type]
            delta_target = graph.edata["delta_target"][ss_type]
            mask = graph.edata["mask"][ss_type]
            logits, delta_logits = self.model(graph)
            operation_loss = F.binary_cross_entropy_with_logits(logits, gamma_target)
            service_loss = F.cross_entropy(delta_logits[mask.bool()],
                                        delta_target[mask.bool()])
            loss = operation_loss + service_loss*service_rate
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())
        return {'train_loss': np.mean(train_loss)}
    
    def test_epoch(self, service_rate):
        test_loss = []
        with torch.no_grad():
            for graph in self.test_loader:
                gamma_target = graph.edata["target"][os_type]
                delta_target = graph.edata["delta_target"][ss_type]
                mask = graph.edata["mask"][ss_type]
                logits, delta_logits = self.model(graph)
                operation_loss = F.binary_cross_entropy_with_logits(logits, gamma_target)
                service_loss = F.cross_entropy(delta_logits[mask.bool()],
                                            delta_target[mask.bool()])
                loss = operation_loss + service_loss*service_rate
                test_loss.append(loss.item())
        return {'test_loss': np.mean(test_loss)}
    
    def update_metrics(self, **to_update):
        wandb.log(to_update)
    
    def run_experiment(self, num_epoch=1000, validation_rate=75, service_rate=0.5,
                       scheduler_rate=30):
        wandb.config.update({
            'num_epoch':num_epoch,
            'validation_rate':validation_rate,
            'scheduler_rate':scheduler_rate,
            'service_rate':service_rate,
        })
        for epoch in trange(num_epoch):
            train_update = self.train_epoch(service_rate=service_rate)
            test_update = self.test_epoch(service_rate=service_rate)
            # collect validation info
            if (epoch + 1) % validation_rate == 0 or epoch==0 or epoch+1==num_epoch:
                obj_train = validate_objective(
                    self.model, self.train_dataset, 'train'
                )
                obj_test = validate_objective(
                    self.model, self.test_dataset, 'test'
                )

                self.update_metrics(
                    **obj_train, **obj_test,
                    epoch=epoch
                )
            # Update scheduler if exist
            if self.scheduler:
                if (epoch+1) % scheduler_rate == 0:
                    self.scheduler.step()
            # Take lr
                step_lr = self.scheduler.get_last_lr()[0]
            else:
                step_lr = self.optim.param_groups[0]["lr"]
            # update general metrics
            self.update_metrics(
                **train_update, **test_update,
                lr = step_lr,
                epoch=epoch
            )