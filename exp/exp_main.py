from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from data_provider.data_loader import Dataset_Custom
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import wandb
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

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
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
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        print(path)
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({'epoch': epoch+1, 'Train Loss': train_loss, 'Val Loss': vali_loss})    
            #loss1.append(train_loss)
            #loss2.append(vali_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val_Loss:{3:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,test_loss))#, test_loss)) Test Loss: {3:.2f}
            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('/home/mail12/Nerf-diffusion/covid_19/FINAL/trans_patchtst/brazil_final/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = '/home/mail12/Nerf-diffusion/covid_19/FINAL/trans_patchtst/brazil_final/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
            
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #print(batch_y[0])
                #rint(dec_inp[0])
                #exit()
                # encoder - decoder
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                #print(pred.shape)
                if self.args.features =='MS':
                    k = []
                    pol = []
                    #print(len(pred))
                    for i in range(len(pred)):
                        p_1 = pred[i]
                        t_1 = true[i]
                        pred_6 =test_data[i][1][-4:] 
                        x1221 = p_1.ravel()
                        pred_6[:,-1] = x1221
                        k.append(pred_6)
                        true_6 =test_data[i][1][-4:] 
                        y1221 = t_1.ravel()
                        true_6[:,-1] = y1221
                        pol.append(true_6)
                    k = np.array(k)
                    k  = k.reshape(4,4)  
                    pol = np.array(pol)
                    pol  = pol.reshape(4,4)  
                    p1 = test_data.inverse_transform(k)#_inversetransform(self.args,pred)
                    y1 = test_data.inverse_transform(pol)
                    p1 = p1.reshape((1,4,4))
                    y1 = y1.reshape((1,4,4))
                    #print(p1)
                    #p1  = p1[:,-1]
                    #y1 = y1[:,-1]
                else:
                    p1 = test_data.inverse_transform(pred.reshape(-1,1))#_inversetransform(self.args,pred)
                    y1 = test_data.inverse_transform(true.reshape(-1,1))
                    p1 = p1.reshape((64,4,1))
                    y1 = y1.reshape((64,4,1))

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                input = batch_x.detach().cpu().numpy()    
                #print(input.shape)
                #x1 = test_data.inverse_transform(input.reshape(640,4))
                #x1 = x1.reshape((64,10,4))
                #print(x1[0,:,-1])
                #print(y1.shape)
                #exit()
                # for ind in range(len(p1)):
                #    gt = np.concatenate((x1[ind, :, -1], y1[ind, :, -1]), axis=0)
                #    pd = np.concatenate((x1[ind, :, -1], p1[ind, :, -1]), axis=0)
                #    visual(gt, pd, os.path.join(folder_path, str(ind) + '.png'))
        #print(y1[-1])
        #print(p1[-1])
        #exit()
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        #print("Preds:- ",preds.shape)
        #print("Truth:-",trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        #print("Preds:- ",preds.shape)

        # result save
        #folder_path = '/home/mail12/Nerf-diffusion/covid_19/FINAL/TRANSFORMER/results_india_MS_500/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.args.features =='MS':
            k1= []
            pol1 = []
            #print(len(pred))
            for i in range(len(preds)):
                p_1 = preds[i]
                t_1 = trues[i]
                pred_6 =test_data[i][1][-4:] 
                x1221 = p_1.ravel()
                pred_6[:,-1] = x1221
                k1.append(pred_6)
                true_6 =test_data[i][1][-4:] 
                y1221 = t_1.ravel()
                true_6[:,-1] = y1221
                pol1.append(true_6)
            k1 = np.array(k1)
            print(k1.shape)
            k1  = k1.reshape(280,4)  
            pol1 = np.array(pol1)
            pol1  = pol1.reshape(280,4)  
            p11 = test_data.inverse_transform(k1)#_inversetransform(self.args,pred)
            y11 = test_data.inverse_transform(pol1)
            p11 = p11.reshape((70,4,4))
            y11 = y11.reshape((70,4,4))
            #p11 = p11[:,-1]
            #y11 = y11[:,-1]
            
        else:
            p11 = test_data.inverse_transform(preds.reshape(-1,1))#_inversetransform(self.args,pred)
            y11 = test_data.inverse_transform(trues.reshape(-1,1)) 
                
            p11 = p11.reshape(64,4,1)
            y11 = y11.reshape(64,4,1)
            
        mae, mse, rmse, mape, mspe, rse, corr, r2score = metric(p11[:,-1], y11[:,-1])
        print('mse:{}, mae:{}, mape:{}, r2:{}'.format(mse, mae, mape, r2score))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape:{},rscore:{}'.format(mse, mae, mape,r2score))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, r2score]))
        np.save(folder_path + 'pred.npy', p11)
        np.save(folder_path + 'true.npy', y11)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = "/home/mail12/Nerf-diffusion/covid_19/FINAL/TRANSFORMER/checkpoints/brazil_S/brazil_10_4_PatchTST_custom_ftS_sl10_ll4_pl4_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_1" + '/' +'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        true = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
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
    
        true = np.array(true)
        preds = np.array(preds)
        print("preds shape:- ",preds.shape)
        print("true shape:- ",true.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        true = true.reshape(-1,true.shape[-2],true.shape[-1])
        inputx= batch_x.reshape(-1,batch_x.shape[-2],batch_x.shape[-1])

        #k = []
        #pol = []
        #for i in range(len(preds)):
        #    p_1 = preds[i]
        #    #t_1 = true[i]
        #    pred_6 =pred_data[i][1][-4:] 
        #    x1221 = p_1.ravel()
        #    pred_6[:,-1] = x1221
        #    k.append(pred_6)
        #k = np.array(k)
        #k  = k.reshape(4,6)  
        #pol = np.array(pol)
        #pol  = pol.reshape(256,6)  

        p11 = pred_data.inverse_transform(preds.reshape(-1,1))
        y11 = pred_data.inverse_transform(true.reshape(-1,1))


        mae_error = np.mean(np.abs(p11- y11))
        mape_error = np.mean(np.abs((p11 - y11) / y11))*100


        print("MAE: - ",mae_error)
        print("MAPE: - ",mape_error)
        plt.plot(p11,label='pred',color='red')
        plt.plot(y11,label='truth',color='green')
        plt.legend()
        plt.ylabel("cases")
        plt.xlabel('days')
        plt.savefig("prediction_india.png")
        plt.close()
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
