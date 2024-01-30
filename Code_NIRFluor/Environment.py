import pandas as pd
import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, Linear, ReLU, Dropout
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_geometric.nn import global_add_pool as gsp, global_mean_pool as gmp, global_max_pool as gap
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import AllChem, Crippen, Lipinski, Draw
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx,:])
        y = self.labels[idx,:]
        return x, y

def set_seed(seed):
    #https://blog.csdn.net/yyywxk/article/details/121606566
    import numpy as np
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def Initlize(model, init_type):
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == 'constant':
                nn.init.constant_(p, 0)
            if init_type == 'ones':
                nn.init.ones_(p)
            if init_type == 'zeros':
                    nn.init.zeros_(p)
            if init_type == 'eye':
                nn.init.eye_(p)
            if init_type == 'orthogonal':
                nn.init.orthogonal_(p)
            if init_type == 'sparse':
                nn.init.sparse_(p)
            if init_type == 'xavier_uniform':       #sigmoid、tanh
                nn.init.xavier_uniform_(p)
            elif init_type == 'xavier_normal':   #sigmoid、tanh
                nn.init.xavier_normal_(p)
            elif init_type == 'kaiming_uniform':    #relu and related
                nn.init.kaiming_uniform_(p)
            elif init_type == 'kaiming_normal':     #relu and related
                nn.init.kaiming_normal_(p)
            #elif init_type == 'small_normal_init':
            #    ModelInit.xavier_normal_small_init_(p)
            #elif init_type == 'small_uniform_init':
            #    ModelInit.xavier_uniform_small_init_(p)
    return model

def one_hot_encoding_unk(value, known_list):
    encoding = [0] * (len(known_list) + 1)
    index = known_list.index(value) if value in known_list else -1
    encoding[index] = 1
    return encoding

class featurization_parameters:
    def __init__(self):
        self.max_atomic_num = 100
        self.atom_features = {'atomic_num': list(range(self.max_atomic_num)),
                              'total_degree': [0, 1, 2, 3, 4, 5],
                              'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
                              'total_numHs': [0, 1, 2, 3, 4],
                              'hybridization': [Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3,
                                                Chem.rdchem.HybridizationType.SP3D,
                                                Chem.rdchem.HybridizationType.SP3D2]}

        self.atom_fdim = sum(len(known_list) + 1 for known_list in self.atom_features.values()) + 3
        self.bond_fdim = 6


feature_params = featurization_parameters()

def atom_features(atom: Chem.rdchem.Atom):
    if atom is None:
        atom_feature_vector  = [0] * feature_params.atom_fdim
    else:
        atom_feature_vector  = one_hot_encoding_unk(atom.GetAtomicNum() - 1, feature_params.atom_features['atomic_num']) + \
            one_hot_encoding_unk(atom.GetTotalDegree(), feature_params.atom_features['total_degree']) + \
            one_hot_encoding_unk(atom.GetFormalCharge(), feature_params.atom_features['formal_charge']) + \
            one_hot_encoding_unk(int(atom.GetTotalNumHs()), feature_params.atom_features['total_numHs']) + \
            one_hot_encoding_unk(int(atom.GetHybridization()), feature_params.atom_features['hybridization']) + \
            [1 if atom.IsInRing()else 0]+ \
            [1 if atom.GetIsAromatic() else 0]+\
            [atom.GetMass() * 0.01]
    return atom_feature_vector

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        bond_feature_vector  = [0] * feature_params.bond_fdim
    else:
        bt = bond.GetBondType()
        bond_feature_vector  = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return bond_feature_vector


def process_single_SMILES(data_row):
    smiles = data_row['Canonical SMILES']
    mol = Chem.MolFromSmiles(smiles)

    xs = []
    for atom in mol.GetAtoms():
        x = atom_features(atom)
        xs.append(x)
    x = torch.tensor(xs)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs).view(-1, 6)

    y_values = data_row.iloc[1:7].values.astype(float)
    y_values = y_values.reshape(1, -1)
    y = torch.tensor(y_values, dtype=torch.float)

    solvent_descriptors = data_row.iloc[7:18].values.astype(float)
    solvent_descriptors = solvent_descriptors.reshape(1, -1)
    solvent_descriptors = torch.tensor(solvent_descriptors, dtype=torch.float)

    mol_fingerprints = data_row.iloc[18:].values.astype(float)
    mol_fingerprints = mol_fingerprints.reshape(1, -1)
    mol_fingerprints = torch.tensor(mol_fingerprints, dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, solvent_descriptors= \
        solvent_descriptors, mol_fingerprints=mol_fingerprints, smiles=smiles)
    return data


def SMILES_data_process(dataset):
    processed_data = []
    for index, row in dataset.iterrows():
        processed = process_single_SMILES(row)
        processed_data.append(processed)
    return processed_data


def process_single_SMILES_ST_GCN(data_row):
    smiles = data_row['Canonical SMILES']
    mol = Chem.MolFromSmiles(smiles)
    
    xs = []
    for atom in mol.GetAtoms():
        x = atom_features(atom)
        xs.append(x)    
    x = torch.tensor(xs)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = bond_features(bond)
        edge_indices += [[i,j],[j,i]]
        edge_attrs += [e, e]
        
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs).view(-1, 6)            

    y_values = float(data_row.iloc[1])
    y = torch.tensor(y_values, dtype=torch.float)
   
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    return data

def SMILES_data_process_ST_GCN(dataset):
    processed_data = []
    for index, row in dataset.iterrows():
        processed = process_single_SMILES_ST_GCN(row)
        processed_data.append(processed)
    return processed_data


def model_optimization_MT_DNN(model, optimizer, loss_function, train_loader, val_loader, early_stopping, epochs):
    train_loss = []
    validation_loss = []
    best_epoch = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.type(torch.float32).to(DEVICE), y.type(torch.float32).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            label = y

            mask = ~torch.isnan(label)  # True: not NAN, False: NAN
            # masked_output = torch.where(mask, outputs, torch.zeros_like(outputs))
            # masked_y = torch.where(mask, label, torch.zeros_like(label))

            # loss = loss_function(masked_output.float(), masked_y.float())
            loss = loss_function(outputs[mask].float(), label[mask].float())  # only calculate the value equal to True
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {average_loss}')

        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.type(torch.float32).to(DEVICE), y_val.type(torch.float32).to(DEVICE)
                val_outputs = model(x_val)
                val_label = y_val

                val_mask = ~torch.isnan(val_label)
                # val_masked_output = torch.where(val_mask, val_outputs, torch.zeros_like(val_outputs))
                # val_masked_y = torch.where(val_mask, val_label, torch.zeros_like(val_label))

                # val_loss = loss_function(val_masked_output.float(), val_masked_y.float())
                val_loss = loss_function(val_outputs[val_mask].float(), val_label[val_mask].float())
                val_total_loss += val_loss.item()

        average_val_loss = val_total_loss / len(val_loader)
        validation_loss.append(average_val_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {average_val_loss}')

        if epoch > 0:
            early_stopping(average_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                best_epoch = early_stopping.best_epoch  # Update the best_epoch value
                break
    print(f"Best model was saved at epoch: {best_epoch}")

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, epoch + 2), train_loss, "r-")
    plt.plot(range(1, epoch + 2), validation_loss, "g-")
    plt.show()
    return train_loss, validation_loss


def model_optimization_GCN(model, optimizer, loss_function, train_loader, val_loader, early_stopping, epochs):
    train_loss = []
    validation_loss = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch)
            label = batch.y

            mask = ~torch.isnan(label)  # True: not NAN, False: NAN
            loss = loss_function(outputs[mask].float(), label[mask].float())  # only calculate the value equal to True
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {average_loss}')

        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(DEVICE)
                val_outputs = model(val_batch)
                val_label = val_batch.y

                val_mask = ~torch.isnan(val_label)
                val_loss = loss_function(val_outputs[val_mask].float(), val_label[val_mask].float())
                val_total_loss += val_loss.item()

        average_val_loss = val_total_loss / len(val_loader)
        validation_loss.append(average_val_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {average_val_loss}')

        if epoch > 0:
            early_stopping(average_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                best_epoch = early_stopping.best_epoch  # Update the best_epoch value
                break
    print(f"Best model was saved at epoch: {best_epoch + 1}")

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, epoch + 2), train_loss, "r-")
    plt.plot(range(1, epoch + 2), validation_loss, "g-")
    plt.show()
    return train_loss, validation_loss



def model_optimization_ST_GCN(model, optimizer, loss_function, train_loader, val_loader, early_stopping, epochs):
    train_loss = []
    validation_loss = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.batch)
            label = data.y
            label = label.unsqueeze(1)
        
            loss = loss_function(outputs.float(), label.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        average_loss = total_loss / len(train_loader)
        train_loss.append(average_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {average_loss}')

        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(DEVICE)
                val_outputs = model(val_data.x, val_data.edge_index, val_data.batch)
                val_label = val_data.y
                val_label = val_label.unsqueeze(1)
                
                val_loss = loss_function(val_outputs.float(), val_label.float())
                val_total_loss += val_loss.item()
        
        average_val_loss = val_total_loss / len(val_loader)
        validation_loss.append(average_val_loss)
        print(f'Epoch {epoch + 1}, Validation Loss: {average_val_loss}')

        if epoch > 0:
            early_stopping(average_val_loss, model, epoch) 
            if early_stopping.early_stop:
                print(f'Early stopping at epoch {epoch + 1}')
                best_epoch = early_stopping.best_epoch  # Update the best_epoch value
                break
    print(f"Best model was saved at epoch: {best_epoch+1}")

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,epoch+2),train_loss,"r-")
    plt.plot(range(1,epoch+2),validation_loss,"g-")
    plt.show()
    return train_loss, validation_loss, best_epoch


def validation_MT_DNN(model, validation_loader, DEVICE):
    loss_function = MSELoss()
    model.eval()
    eval_loss = 0
    validation_loss = []
    predictions = []
    true_labels = []
    mask_total = []

    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.type(torch.float32).to(DEVICE), y.type(torch.float32).to(DEVICE)
            outputs = model(x)
            label = y
            mask = ~torch.isnan(label)

            predictions.extend(outputs.tolist())
            true_labels.extend(label.tolist())
            mask_total.extend(mask.tolist())

            # masked_output = torch.where(mask, outputs, torch.zeros_like(outputs))
            # masked_y = torch.where(mask, label, torch.zeros_like(label))

            # loss = loss_function(masked_output, masked_y)
            loss = loss_function(outputs[mask], label[mask])

            eval_loss += loss.item()

    eval_loss /= len(validation_loader)
    validation_loss.append(eval_loss)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    mask_total = np.array(mask_total)

    print('validation finished!')
    print('validation_loss: {:.4f}'.format(eval_loss))
    return predictions, true_labels, mask_total, validation_loss

def validation_GCN(model, validation_loader, DEVICE):
    loss_function = MSELoss()
    model.eval()
    eval_loss = 0
    validation_loss = []
    predictions = []
    true_labels = []
    mask_total = []

    with torch.no_grad():
        for batch in validation_loader:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            label = batch.y
            mask = ~torch.isnan(label)

            predictions.extend(outputs.tolist())
            true_labels.extend(label.tolist())
            mask_total.extend(mask.tolist())

            loss = loss_function(outputs[mask], label[mask])
            eval_loss += loss.item()

    eval_loss /= len(validation_loader)
    validation_loss.append(eval_loss)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    mask_total = np.array(mask_total)

    print('validation finished!')
    print('validation_loss: {:.4f}'.format(eval_loss))
    return predictions, true_labels, mask_total, validation_loss


def validation_ST_GCN(model, validation_loader, DEVICE):
    loss_function = MSELoss()
    model.eval()
    eval_loss = 0
    validation_loss=[]
    predictions = []
    true_labels = []
    
    with torch.no_grad():    
        for data in validation_loader:
            data = data.to(DEVICE)
            outputs = model(data.x, data.edge_index, data.batch)
            label = data.y
            label = label.unsqueeze(1)
            
            predictions.extend(outputs.tolist())
            true_labels.extend(label.tolist())
            
            loss = loss_function(outputs,label)
            eval_loss += loss.item()
    
    eval_loss /= len(validation_loader)
    validation_loss.append(eval_loss)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    print('validation finished!')   
    print('validation_loss: {:.4f}'.format(eval_loss))
    return predictions, true_labels, validation_loss


class MT_DNN(nn.Module):
    def __init__(self, num_properties=6):
        super(MT_DNN, self).__init__()
        self.fc_layers = nn.Sequential(
            Linear(2076, 512),
            ReLU(),
            # Dropout(p=0.1),
            # Linear(512, 256),
            # ReLU(),
            # Linear(256, 128),
            # ReLU()
        )

        self.property_heads = nn.ModuleList([nn.Sequential(Linear(512, 32), ReLU(),
                                                           Linear(32, 1)) for _ in range(num_properties)])

    def forward(self, x):
        x = self.fc_layers(x)
        outputs = [head(x) for head in self.property_heads]
        return torch.cat(outputs, dim=1)

            
class MT_FinGCN(torch.nn.Module):
    def __init__(self, dropout=0.3):
        super(MT_FinGCN, self).__init__()
        self.in_channels = 131
        self.hidden_channels = 256
        self.out_channels = 64
        self.num_properties = 6
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.out_channels)

        self.dropout = Dropout(p=dropout)
        self.linear1 = Linear(self.out_channels + 11 + 2065, 4096)
        self.linear2 = Linear(4096, 512)
        self.linear3 = Linear(512, 128)
        self.property_heads = nn.ModuleList([nn.Sequential(
            Linear(128, 32), ReLU(), Dropout(p=dropout),
            Linear(32, 1)) for _ in range(self.num_properties)])

    def forward(self, data):
        x, edge_index, edge_attr, batch_index, solvent_descriptors, mol_fingerprints = \
            data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_descriptors, data.mol_fingerprints

        x = F.relu(self.conv1(x.float(), edge_index))
        x = self.conv2(x, edge_index)
        x = gsp(x, batch_index)

        x = torch.cat(
            [x, solvent_descriptors.reshape(data.num_graphs, 11), mol_fingerprints.reshape(data.num_graphs, 2065)],
            dim=1)

        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        outputs = [head(x) for head in self.property_heads]
        return torch.cat(outputs, dim=1)

class MT_GCN(torch.nn.Module):
    def __init__(self,dropout=0.3):
        super(MT_GCN, self).__init__()
        self.in_channels = 131
        self.hidden_channels = 128
        self.out_channels = 64
        self.num_properties = 6
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.out_channels)
        
        self.linear1 = Linear(self.out_channels+11,2048)
        self.linear2 = Linear(2048, 1024)
        self.dropout = Dropout(p=dropout)
        self.linear3 = Linear(1024, 128)
        self.property_heads = nn.ModuleList([nn.Sequential(
            Linear(128, 32),ReLU(),Dropout(p=dropout), 
            Linear(32, 1)) for _ in range(self.num_properties)])

    def forward(self, data):
        x, edge_index, edge_attr, batch_index, solvent_descriptors, mol_fingerprints = \
        data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_descriptors, data.mol_fingerprints
        
        x = F.relu(self.conv1(x.float(), edge_index))
        x = self.conv2(x, edge_index)
        x = gsp(x,batch_index)
        
        x = torch.cat([x, solvent_descriptors.reshape(data.num_graphs,11)],dim=1)
      
        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        outputs = [head(x) for head in self.property_heads]
        return torch.cat(outputs, dim=1)

class MT_GCN_non_SF(torch.nn.Module):
    def __init__(self,dropout=0.3):
        super(MT_GCN_non_SF, self).__init__()
        self.in_channels = 131
        self.hidden_channels = 128
        self.out_channels = 64
        self.num_properties = 6
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.out_channels)
        
        self.linear1 = Linear(self.out_channels,2048)
        self.linear2 = Linear(2048, 1024)
        self.dropout = Dropout(p=dropout)
        self.linear3 = Linear(1024, 128)
        self.property_heads = nn.ModuleList([nn.Sequential(
            Linear(128, 32),ReLU(),Dropout(p=dropout), 
            Linear(32, 1)) for _ in range(self.num_properties)])

    def forward(self, data):
        x, edge_index, edge_attr, batch_index, solvent_descriptors, mol_fingerprints = \
        data.x, data.edge_index, data.edge_attr, data.batch, data.solvent_descriptors, data.mol_fingerprints
        
        x = F.relu(self.conv1(x.float(), edge_index))
        x = self.conv2(x, edge_index)
        x = gsp(x,batch_index)
      
        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        outputs = [head(x) for head in self.property_heads]
        return torch.cat(outputs, dim=1)

class ST_GCN_non_SF(torch.nn.Module):
    def __init__(self,dropout=0.3):
        super(ST_GCN_non_SF, self).__init__()
        self.in_channels = 131
        self.hidden_channels = 128
        self.out_channels = 64
        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.out_channels)
        
        self.dropout = Dropout(p=dropout)
        self.linear1 = Linear(self.out_channels,2048)
        self.linear2 = Linear(2048, 1024)
        self.linear3 = Linear(1024, 128)
        self.linear4 = Linear(128,32)
        self.linear5 = Linear(32,1)
        
    def forward(self, x, edge_index, batch=None):
        if batch is not None:
            batch_index = batch
        
        x = F.relu(self.conv1(x.float(), edge_index))
        x = self.conv2(x, edge_index)
        x = gsp(x,batch)
      
        x = F.relu(self.linear1(x.float()))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        x = F.relu(self.linear4(x))
        x = self.dropout(x)
        x = self.linear5(x)
        return x
            

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='./Result_DNN_pred/DNN_best_model.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_epoch = 0  # Track the epoch of the best model

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch  # Update the epoch of the best model
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch  # Update the epoch of the best model
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_train_metrics(predictions, true_labels, masks):
    train_metrics = {}

    for i in range(predictions.shape[1]):
        masked_true = true_labels[:, i][masks[:, i]]
        masked_pred = predictions[:, i][masks[:, i]]

        R2 = r2_score(masked_true, masked_pred)
        MSE = mean_squared_error(masked_true, masked_pred)
        RMSE = np.sqrt(MSE)

        train_metrics[f'R2_{i}'] = R2
        train_metrics[f'RMSE_{i}'] = RMSE
    return train_metrics


def calculate_val_metrics(predictions, true_labels, masks):
    val_metrics = {}

    for i in range(predictions.shape[1]-1):
        masked_true = true_labels[:, i][masks[:, i]]
        masked_pred = predictions[:, i][masks[:, i]]

        MSE = mean_squared_error(masked_true, masked_pred)
        RMSE = np.sqrt(MSE)
        val_metrics[f'RMSE_{i}'] = RMSE
    return val_metrics
        

def save_metrics_to_txt(metrics, file_name):
    with open(file_name, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")