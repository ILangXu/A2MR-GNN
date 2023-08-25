import torch
from model import GINNet as DeepMice
from dataset import get_gin_dataloader
from utils import set_data_device
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
import numpy as np
from torch.nn import Sequential, Linear, ReLU

class Predictor:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.model = DeepMice()
        self.model = self.model.to(device)
        self.test_path = 'data/mice_features/v2016/all_5A/'
        self.label_path = 'data/mice_features/total_labels.pkl'
        self.testloader = get_gin_dataloader(self.test_path,
                                             self.label_path,
                                             batch_size=64,
                                             phase='test')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self):
        y_pred = []
        y_label = []
        x_graph_list = []
        results = pd.DataFrame(columns=['pred', 'label'])
        with torch.no_grad():
            for i, (x, y) in enumerate(self.testloader):
                x, y = set_data_device((x, y), self.device)
                score, x_graph = self.model(x)
                logits = torch.squeeze(score).detach().cpu().numpy()
                label_ids = y.to('cpu').numpy()
                temp_df = pd.DataFrame({'pred': logits, 'label': label_ids})
                results = pd.concat([results, temp_df], ignore_index=True)
                y_label = y_label + label_ids.flatten().tolist()
                y_pred = y_pred + logits.flatten().tolist()
                x_graph = x_graph.to('cpu')
                x_graph_list.append(x_graph)
                if (i==3):
                    break

        x_graph_array = np.concatenate(x_graph_list, axis=0)
        x_graph_df = pd.DataFrame(x_graph_array, columns=[f'feature_{j}' for j in range(x_graph_array.shape[1])])
        x_graph_df.to_csv('x_graph.csv', index=False)
        y_label_df = pd.DataFrame({'label': y_label})
        y_label_df.to_csv('y_label.csv', index=False)

        results.to_csv(f'results_test.csv', index=False)

        mse = mean_squared_error(y_label, y_pred)
        pearson = pearsonr(y_label, y_pred)
        ci = concordance_index(y_label, y_pred)
        mae = mean_absolute_error(y_label, y_pred)
        r2 = r2_score(y_label, y_pred)

        return mse, pearson[0], pearson[1], ci, mae, r2, y_pred


if __name__ == '__main__':
    model_path = 'real_best/model_general2016-core2016.pt'  # 指定已保存的模型路径
    predictor = Predictor()
    predictor.load_model(model_path)
    mse, pearson, p_val, ci, mae, r2, predictions = predictor.predict()
    print('MSE:', mse)
    print('Pearson Correlation:', pearson)
    print('p-value:', p_val)
    print('Concordance Index:', ci)
    print('MAE:', mae)
    print('R^2:', r2)
