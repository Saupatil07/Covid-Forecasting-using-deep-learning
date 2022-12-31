from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from data_provider.data_loader import Dataset_Custom
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

class Exp_Main_forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_forecast, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        print(best_model_path)
        preds = []
        true = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                #print(batch_x[0])
                #print(batch_y[0])
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #print(dec_inp[0])
                #exit()
                # encoder - decoder
                #print(batch_x[:,:,-1])
                #print(batch_y[:,:,-1])
                #print(dec_inp[:,:,-1])
                #exit()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

                batch_y = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
                batch_y = batch_y.detach().cpu().numpy()
                true.append(batch_y)
            #exit()
        true = np.array(true)
        preds = np.array(preds)
        #exit()
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        true = true.reshape(-1,true.shape[-2],true.shape[-1])
        inputx= batch_x.reshape(-1,batch_x.shape[-2],batch_x.shape[-1])
        folder_path = '/home/mail12/Nerf-diffusion/covid_19/FINAL/trans_patchtst/brazil_final/brazil_forecast/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print("preds shape:- ",preds.shape)
        print("true shape:- ",true.shape)
        p1 = true[0]
        p2 = preds[0]
        p2 = p2.ravel()
        p1[:,-1] = p2
        y = batch_y
        p11 = pred_data.inverse_transform(p1)#.reshape(-1,1))
        y11 = pred_data.inverse_transform(y[0])#.reshape(-1,1))
        print(p11[:,-1])
        print(y11[:,-1])
        mae_error = np.mean(np.abs(p11[:,-1]- y11[:,-1]))
        mse_error = np.mean((p11[:,-1] - y11[:,-1]) ** 2)
        rmse_error = np.sqrt(mse_error)
        print("RMSE:-",rmse_error)
        print("MAE: - ",mae_error)
        #from sklearn.metrics import r2_score
        #from sklearn.metrics import explained_variance_score
        #print("R2 score:-",explained_variance_score(y11[:,-1],p11[:,-1]))
        #result save
        #folder_path = './results/' + setting + '/'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)

        np.save(folder_path+'real_prediction1.npy', np.array([p11[:,-1],y11[:,-1]]))
        np.save(folder_path+'metrics_pred.npy',np.array([mae_error,rmse_error]))
        # if self.args.features =='MS':
        #     k = []
        #     pol = []
        #     print(len(pred))
        #     for i in range(len(pred)):
        #         p_1 = pred[i]
        #         t_1 = true[i]
        #         pred_6 =pred_data[i][1][-4:] 
        #         x1221 = p_1.ravel()
        #         pred_6[:,-1] = x1221
        #         k.append(pred_6)
        #         true_6 =pred_data[i][1][-4:] 
        #         y1221 = t_1.ravel()
        #         true_6[:,-1] = y1221
        #         pol.append(true_6)
        #     k = np.array(k)
        #     k  = k.reshape(256,6)  
        #     pol = np.array(pol)
        #     pol  = pol.reshape(256,6)  
        #     p1 = pred_data.inverse_transform(k)#_inversetransform(self.args,pred)
        #     y1 = pred_data.inverse_transform(pol)
        #     p1 = p1.reshape((64,4,6))
        #     y1 = y1.reshape((64,4,6))
        # mae_error = np.mean(np.abs(p11[:,-1]- y11[:,-1]))
        # print("MAE: - ",mae_error)
        # # if self.args.features=='MS':
        #     #print(preds[0])
        #     #print(true[0])
        #     #exit()
        #     p11 = pred_data.inverse_transform(preds[0])#.reshape(28,6))#.reshape(-1,1))
        #     y11 = pred_data.inverse_transform(true[0])#.reshape(28,6))#.reshape(-1,1))
        #     #print(p11[:,-1][:5])#.shape)
        #     #print(y11[:,-1][:5])
        #     #exit()#.shape)
        #     #print(np.sum(np.abs(p11[:,-1]-y11[:,-1]))/4)
        #     mae_error = np.mean(np.abs(p11[:,-1][:5]- y11[:,-1][:5]))
        #     mse_error = np.mean((p11[:,-1][:5] - y11[:,-1][:5]) ** 2)
        #     rmse_error = np.sqrt(mse_error)
        #     #mape_error = np.mean(np.abs((p11 - y11) / y11))*100
        #     #print(p11.shape)
        #     #from sklearn.metrics import r2_score
        #     #print("R2 score:-",r2_score(y11,p11))
        #     print("MAE: - ",mae_error)
        #     print("MSE: - ",mse_error)
        #     print("RMSE:- ",rmse_error)
        #     #exit()
        #     plt.plot(p11[:,-1],label='pred',color='red')
        #     plt.plot(y11[:,-1],label='truth',color='green')
        #     plt.legend()
        #     plt.ylabel("cases")
        #     plt.xlabel('days')
        #     plt.savefig(folder_path+"prediction.png")
        #     plt.close()
            
        #     # result save
        #     #folder_path = './results/' + setting + '/'
        #     #if not os.path.exists(folder_path):
        #     #    os.makedirs(folder_path)

        #     np.save(folder_path + 'real_prediction.npy', np.array([p11[:,-1],y11[:,-1]]))
        #     np.save(folder_path + 'metrics.npy',mae_error)
        # else:
        #     print("in else")
        #     p11 = pred_data.inverse_transform(preds.reshape(-1,1))
        #     y11 = pred_data.inverse_transform(true.reshape(-1,1))
        #     print(p11)
        #     print(y11)
        #     print(p11.shape)
        #     mae_error = np.mean(np.abs(p11- y11))
        #     mse_error = np.mean((p11 - y11) ** 2)
        #     rmse_error = np.sqrt(mse_error)
        #     #mape_error = np.mean(np.abs((p11 - y11) / y11))*100
        #     #print(p11.shape)
        #     #from sklearn.metrics import r2_score
        #     #print("R2 score:-",r2_score(y11,p11))
        #     print("MAE: - ",mae_error)
        #     print("MSE: - ",mse_error)
        #     print("RMSE:- ",rmse_error)
        #     # #exit()
        #     # plt.plot(p11[:,-1],label='pred',color='red')
        #     # plt.plot(y11[:,-1],label='truth',color='green')
        #     # plt.legend()
        #     # plt.ylabel("cases")
        #     # plt.xlabel('days')
        #     # plt.savefig(folder_path+"prediction.png")
        #     # plt.close()
            
        #     # # result save
        #     # #folder_path = './results/' + setting + '/'
        #     # #if not os.path.exists(folder_path):
        #     # #    os.makedirs(folder_path)

        #     # np.save(folder_path + 'real_prediction.npy', np.array([p11[:,-1],y11[:,-1]]))
        #     # np.save(folder_path + 'metrics.npy',mae_error)
        return mae_error
