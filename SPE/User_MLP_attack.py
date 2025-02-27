import numpy

from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
#
# from catSNN import spikeLayer, SpikeDataset ,load_model, fuse_module
import catCuda
import os

C_level=3
Q_level=2
seed=42
noise=3
quantized_weight=3

def set_seeds(seed=42):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seeds(seed)
    
def quantize_to_bit_(x, nbit):
    x = (1-2.0**(1-nbit))*x
    x = torch.clamp(x,-1,1)
    x = torch.round(torch.div(x, 2.0**(1-nbit)))
    return x

def create_spike_input_cuda(input,T, theta):
    spikes_data = [input for _ in range(T)]
    out = torch.stack(spikes_data, dim=-1).type(torch.FloatTensor).cuda() #float
    out = catCuda.getSpikes(out, theta-0.0001)
    return out

class CalculateLoss(torch.nn.Module):
    def __init__(self, q_level,c_max):
        super(CalculateLoss, self).__init__()
        self.q_level = q_level
        self.max_value=c_max

    def forward(self, x):
        q_level = self.q_level
        c_max = self.max_value
        # Simplifying calculations by combining conditions and using in-place operations where possible
        Safe_zero_mask = (x <= 0)
        Safe_one_mask =  (x >= c_max + 0.5/q_level)

        x_scaled = x * q_level
        k = 2 * torch.round(x_scaled - 0.5 - 1e-5) + 1  # Finds the nearest odd integer to x_scaled
        seq_val = (k * 0.5) / q_level

        # Using torch.where to combine operations and reduce memory usage
        seq_val = torch.where((x >= 0) &(x <= 1/q_level), 0, seq_val)
        seq_val = torch.where(Safe_zero_mask, x, seq_val)
        seq_val = torch.where(Safe_one_mask, x, seq_val)
        x = torch.where((x >= 0) &(x <= 1/q_level), 0.5*x, x)
        # Loss calculation
        act_loss =  torch.sum(torch.abs(x - seq_val)) 

        #act_loss = (0.5 / q_level) ** 2 * torch.mean(torch.pow((torch.abs(x - seq_val) + 1e-10) / (0.5 / q_level), 0.5))

        return act_loss

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=100):
        ctx.constant = constant
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        return C_level * F.hardtanh(grad_output/C_level), None
        # return F.hardtanh(grad_output), None
Quantization_ = Quantization.apply


class MLP(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, output_size=1):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
        self.cal = CalculateLoss(Q_level,C_level)

    def forward(self, x):
        # print("-------x",x,x.size())
        x = torch.clamp(x,min=0, max=C_level)
        x = Quantization_(x,Q_level)
        x = self.fc1(x)
        y1 = self.cal(x)
        # x = self.relu(x)
        x = torch.clamp(x,min=0, max=C_level)
        x = Quantization_(x,Q_level)
        x = self.fc2(x)
        y2 = self.cal(x)
        # x = self.relu(x)
        x = torch.clamp(x,min=0, max=C_level)
        x = Quantization_(x,Q_level)
        x = self.fc3(x)
        y3 = self.cal(x)
        x = self.sigmoid(x)
        y = y2 + y3 + y1
        return x, y

class MLP_snn(nn.Module):
    def __init__(self, input_size=132, hidden_size=128, output_size=1):
        super(MLP_snn, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, output_size)
        # self.fc4 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, factor1, factor2, factor3):
        theta = (1-2.0**(1-quantized_weight)) * C_level * 2.0**(quantized_weight-1)
        # print("-------x",x,x.size())
        x1_spike = create_spike_input_cuda(x, C_level*Q_level, C_level)
        x1 = x1_spike.transpose(-2, -1)
        
        # factor = torch.max(torch.abs(self.fc1.weight))
        # if factor<torch.max(torch.abs(self.fc1.bias)):
        #     factor = torch.max(torch.abs(self.fc1.bias))

        x1 = F.linear(x1, C_level*self.fc1.weight/Q_level, self.fc1.bias/Q_level)
        x = torch.sum(x1, dim=2)/C_level
        # x = self.fc1(x) 
        x1_spike = create_spike_input_cuda(x, C_level*Q_level, theta/factor1)

        x1 = x1_spike.transpose(-2, -1)
        # x = torch.sum(x1, dim=3)
        x1 = F.linear(x1, C_level*self.fc2.weight/Q_level, self.fc2.bias/Q_level)
        x = torch.sum(x1, dim=2)/C_level
        x = create_spike_input_cuda(x, C_level*Q_level, theta/factor2)
        total_mean = torch.mean(torch.cat([x.flatten(), x1_spike.flatten()]))
        with open('total_mean_values.txt', 'a') as file:
            file.write(f"{total_mean.item()}\n")
        x = x.transpose(-2, -1)
        x = F.linear(x, C_level * self.fc3.weight/Q_level, self.fc3.bias/Q_level)
        x = torch.sum(x, dim=2)/C_level
        x = self.sigmoid(x)
        return x

class ModelTrainer:
    def __init__(self, model_type='mlp', input_size=132, hidden_size=128, num_channels=1, num_epochs=50, batch_size=1, lr=0.01, user=0, loadpath=None, savepath='model/cq'):
        self.model_type = model_type.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.user = user
        self.loadpath = loadpath
        self.savepath = savepath
        self.model_load_path = f'{self.loadpath}/{self.user}.pth'
        self.model_save_path = f'{self.savepath}/{self.user}.pth'
        self.log_file_path = f'{self.savepath}/log.txt'

        if self.model_type == 'mlp':
            self.model = MLP(self.input_size, self.hidden_size)

        elif self.model_type == 'cnn':
            self.model = CNN_cq(self.input_size, num_channels)
        else:
            raise ValueError("Unsupported model type. Choose 'mlp' or 'cnn'.")
        self.model = self.model.to(self.device)
        self.snn = MLP_snn(self.input_size, self.hidden_size).to(self.device)
        self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.factor1 = 1
        self.factor2 = 1
        self.factor3 = 1

    def log_training(self, user, f1_score, confusion_mat, acc):
    # self.log_training(self.user, f1, cm,acc)
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(f"Batchsize: {self.batch_size}\n")
            log_file.write(f"LearningRate: {self.lr}\n")
            log_file.write(f"Epochs: {self.num_epochs}\n")
            log_file.write(f"For user: {user}\n")
            log_file.write(f"Accuracy: {acc:.4f}\n")
            log_file.write(f"F1 Score: {f1_score:.4f}\n")
            log_file.write(f"Confusion Matrix:\n{confusion_mat}\n\n")
    
    def orthogonality_loss(self, model, beta):
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if 'weight' in name:  # 只关注包含'weight'的参数
                W = param  # 直接使用参数，因为我们知道这是一个二维的权重矩阵
                WT_W = torch.matmul(W.T, W)  # 计算 W.T * W
                I = torch.eye(WT_W.size(0), device=param.device)
                reg_loss += (WT_W - beta * I).pow(2).sum()
        return reg_loss

    def train_model(self, train_loader):
        self.model.train()
        if self.loadpath!=None:
            self.model.load_state_dict(torch.load(self.model_load_path, map_location=self.device))

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:

                # inputs, labels = inputs.float(), labels.float()
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                self.optimizer.zero_grad()
                outputs, l2 = self.model(inputs)
                outputs = outputs.view(-1)  # Ensure it's a 1D array
                # loss = self.criterion(outputs, labels)
                loss = self.criterion(outputs, labels) + l2 * 1e-4 + 1e-9 * self.orthogonality_loss(self.model,0)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            # print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        state_dict = self.model.state_dict()
        factor1 = max(torch.max(torch.abs(state_dict["fc1.weight"])), torch.max(torch.abs(state_dict["fc1.bias"])))
        factor2 = max(torch.max(torch.abs(state_dict["fc2.weight"])), torch.max(torch.abs(state_dict["fc2.bias"])))
        factor3 = max(torch.max(torch.abs(state_dict["fc3.weight"])), torch.max(torch.abs(state_dict["fc3.bias"])))
        # self.model.state_dict["fc1.weight"] *= 2
        state_dict["fc1.weight"] /= factor1
        state_dict["fc1.weight"] = nn.Parameter(quantize_to_bit_(state_dict["fc1.weight"], quantized_weight))
        state_dict["fc1.bias"]/=factor1
        state_dict["fc1.bias"]= nn.Parameter(quantize_to_bit_(state_dict["fc1.bias"], quantized_weight))
        state_dict["fc2.weight"] /= factor2
        state_dict["fc2.weight"] = nn.Parameter(quantize_to_bit_(state_dict["fc2.weight"], quantized_weight))
        state_dict["fc2.bias"]/=factor2
        state_dict["fc2.bias"]= nn.Parameter(quantize_to_bit_(state_dict["fc2.bias"], quantized_weight))
        state_dict["fc3.weight"] /= factor3
        state_dict["fc3.weight"] = nn.Parameter(quantize_to_bit_(state_dict["fc3.weight"], quantized_weight))
        state_dict["fc3.bias"]/=factor3
        state_dict["fc3.bias"]= nn.Parameter(quantize_to_bit_(state_dict["fc3.bias"], quantized_weight))
        # self.model.load_state_dict(state_dict)
        torch.save(state_dict, 'tmp_ann.pth')
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3

    def evaluate_model(self, test_loader, snn=True, epsilon=0, noise=noise):
        self.model.eval()
        self.snn.eval()
        # torch.save(self.model.state_dict(), 'tmp_ann.pth')
        self.snn.load_state_dict(torch.load('tmp_ann.pth'))
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                if noise==1:
                    inputs+= torch.FloatTensor(inputs.size()).uniform_(-epsilon, epsilon).cuda()
                elif noise==2:
                    inputs += torch.randn(inputs.shape).cuda() * epsilon
                elif noise==3:
                    inputs += torch.poisson(torch.ones_like(inputs)) * epsilon
                
                if snn:
                    # outputs = self.snn(inputs)
                    outputs = self.snn(inputs, self.factor1, self.factor2, self.factor3)
                else:
                    outputs,_ = self.model(inputs)
                
                outputs = outputs.view(-1)  # Ensure outputs are 1D
                preds = (outputs > 0.5).float()  # Calculate pred`icted classes
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_probs.extend(outputs.tolist())

        # Compute metrics
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        # torch.save(self.model.state_dict(), self.model_save_path)
        # print(f"Model saved to {self.model_save_path}")
        # self.log_training(self.user, f1, cm,acc)

        return acc, cm, f1, all_preds, all_probs

    def prepare_data(self, X_train, y_train, X_test, y_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        return train_loader, test_loader

    def run(self, X_train, y_train, X_test, y_test):
        train_loader, test_loader = self.prepare_data(X_train, y_train, X_test, y_test)
        self.train_model(train_loader)
        results = []
        # for epsilon in [0.1 * i for i in range(1, 11)]:
        # for epsilon in numpy.arange(0.02, 0.22, 0.02):
        # for epsilon in numpy.arange(0.05, 0.51, 0.05):
        for epsilon in numpy.arange(0.02, 0.21, 0.02):
            accuracy, confusion_mat, f1_score_val, predictions, probabilities = self.evaluate_model(test_loader, epsilon=epsilon)
            print("Accuracy:", accuracy)
            print("Confusion Matrix:\n", confusion_mat)
            print("F1 Score:", f1_score_val)
            results.append({
                'epsilon': epsilon,
                'predictions': predictions,
                'probabilities': probabilities,
                'confusion_matrix': confusion_mat,
            })

        # # 返回所有epsilon的结果
        return results




def listOfFeatures2Matrix(features):
    '''
    listOfFeatures2Matrix(features)

    This function takes a list of feature matrices as argument and returns a single concatenated feature matrix and the respective class labels.

    ARGUMENTS:
        - features:        a list of feature matrices

    RETURNS:
        - X:            a concatenated matrix of features
        - Y:            a vector of class indeces
    '''

    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)


'''
def normalizeFeatures(features):
    X = numpy.array([])
    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)
'''




def normalizeFeaturesWithoutSMOTE(features):

    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0) + 0.00000000000001;
    STD = numpy.std(X, axis=0) + 0.00000000000001;

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - MEAN) / STD
        features_norm.append(ft)
    return (features_norm, MEAN, STD)


#后smote
def normalizeFeaturesWithSMOTE(features):
    X = numpy.vstack(features)
    Y = numpy.hstack([numpy.full((feat.shape[0],), i) for i, feat in enumerate(features)])

    MEAN = numpy.mean(X, axis=0)
    STD = numpy.std(X, axis=0) + 1e-8
    X_norm = (X - MEAN) / STD
    smote = SMOTE(random_state=42)
    X_res, Y_res = smote.fit_resample(X_norm, Y)

    features_norm = []
    for label in numpy.unique(Y_res):
        ft = X_res[Y_res == label]
        features_norm.append(ft)

    return (features_norm, MEAN, STD)



def normalizeFeatures(trNF, trF):

    if len(trNF)+len(trF)<150:
        print("SMOTE")
        return normalizeFeaturesWithoutSMOTE([trNF, trF])
    else:
        print("NO SMOTE")
        return normalizeFeaturesWithoutSMOTE([trNF, trF])