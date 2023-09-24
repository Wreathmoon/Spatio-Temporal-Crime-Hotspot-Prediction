# Luohao Xu edsml-lx122

import os
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import config
from DataPreLoader import DataPreLoader
from LSTMModel import ConvLSTMModel
from DataPreprocessing import DataPreprocessing

def bceLoss(pred,target,weights=config.BCE_WEIGHTS):
    '''
    Calculate weighted binary cross entropy loss
    
    Inputs - pred: predicted probability scores [batch_size, 1, crime_types, h, w]
             target: binned target values
             weights: binary entropy weights
    '''
    pred = torch.clamp(pred,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(pred) - (1 - target) * weights[0] * torch.log(1 - pred)
    return torch.mean(bce)


def train(train_dl, val_dl, model, optim, scheduler, epochs, batch_size, save, start_epoch, model_save_path):
    '''
    Training loop
    Inputs - train_dl: training data loader
             val_dl: validation data loader
             model: model to be trained
             optim: optimser
             epochs: total number of epochs
             batch_size: batch size
             save: whether to save model checkpoints
             start_epoch: starting epoch number
             model_save_path: path to save trained model
    '''
    optim_name = type(optim).__name__
    writer = SummaryWriter(comment='-optim-({})_lr-({})_bs-({})_threshold-({})_rs-({})-nepoch-({})_weights-({})'
                            .format(optim_name, config.LEARNING_RATE, config.TRAIN_BATCH_SIZE, 
                            config.CLASS_THRESH, config.RANDOM_SEED, config.N_EPOCHS, config.BCE_WEIGHTS))
    
    # add network graph to the tensorboard
    example_input = torch.zeros(config.TRAIN_BATCH_SIZE, config.SEQ_LEN, config.CRIME_TYPE_NUM, config.LAT_GRIDS, config.LON_GRIDS).to(config.DEVICE)
    writer.add_graph(model, (example_input,))
    
    best_model = model
    best_epoch = 0
    best_loss = 10000.0

    for epoch in range(start_epoch,epochs):

        print(f'\n Epoch: {epoch} \n')
        epoch_loss = 0.0
        total = 0
        all_outputs = list()
        all_targets = list()
        model.train()

        for i, (X, Y) in enumerate(tqdm(train_dl, ncols=75)):
            if Y.shape[0] == batch_size:

                pred_scores = model(X)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = bceLoss(pred_scores, Y)                

                optim.zero_grad()
                wce_loss.backward()
                optim.step()
                pred_bin = (pred_scores > config.CLASS_THRESH).float() 
                # all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                # all_targets.append(Y.view(-1,1).detach().cpu().numpy())
                total += Y.shape[0]
                epoch_loss += wce_loss.item()
        scheduler.step()

        # all_outputs = np.concatenate(all_outputs)        
        # all_targets = np.concatenate(all_targets)
        # recall = recall_score(y_pred=all_outputs,y_true=all_targets,average='weighted') ################################ x3333333333
        # precision = precision_score(y_pred=all_outputs,y_true=all_targets,average='weighted') ################################ x3333333333
        # f1score = f1_score(y_pred=all_outputs,y_true=all_targets,average='weighted')

        avg_loss = epoch_loss/total

        # print(f'Train Recall Score: {recall}')
        # print(f'Train Precision Score: {precision}')

        best_model, best_loss, best_epoch, val_f1, val_recall, val_precision, val_avg_loss = validate(val_dl, model, batch_size, epoch, best_loss, best_model, best_epoch, writer)

        print(f'Train Loss: {avg_loss}')
        print(f'Validation Loss: {val_avg_loss}')
        
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Loss/Val', val_avg_loss, epoch)
        # writer.add_scalar('Recall Score/Train', recall, epoch)
        writer.add_scalar('Recall Score/Val', val_recall, epoch)
        # writer.add_scalar('Precision Score/Train', precision, epoch)
        writer.add_scalar('Precision Score/Val', val_precision, epoch)
        # writer.add_scalar('F1 Score/Train', f1score, epoch)
        writer.add_scalar('F1 Score/Val', val_f1, epoch)


        if save:
            print('Saving model')
            checkpoint = {
                'model': model.state_dict(),
                'optim': optim.state_dict,
                'epoch': epoch     
            }
            torch.save(checkpoint,model_save_path+f'/CheckPoint__bs-({config.TRAIN_BATCH_SIZE})_threshold-({config.CLASS_THRESH})_weights-({config.BCE_WEIGHTS}).pt')
    
    writer.close()

    return best_model, best_loss, best_epoch


def validate(dl, model, batch_size, epoch ,best_loss, best_model, best_epoch, writer):
    '''
    Validation loop

    Inputs - dl: validation data loader
             model: trained model
             batch_size: batch size
             epoch: current epoch number
             best_recall: current best recall score
             best model: current best model
             writer: tensorboard summary writer
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    all_outputs = list()
    all_outputs_probs = list()
    all_targets = list()

    with torch.no_grad():
        for i, (X, Y) in enumerate(dl):
            if Y.shape[0] == batch_size:                
                pred_scores = model(X)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = bceLoss(pred_scores, Y)

                if i == 0 or i == 1:
                    pred_bin_sum = (pred_scores > config.CLASS_THRESH).float().sum()
                    print(f'Pred Score Sum - {i}: {pred_bin_sum}')

                pred_bin = (pred_scores > config.CLASS_THRESH).float()
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(Y.view(-1,1).detach().cpu().numpy())
                all_outputs_probs.append(pred_scores.view(-1,1).detach().cpu().numpy())
                total += Y.shape[0]
                epoch_loss += wce_loss.item()
                
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        all_outputs_probs = np.concatenate(all_outputs_probs)
        
        recall = recall_score(y_pred=all_outputs,y_true=all_targets,average='weighted')
        precision = precision_score(y_pred=all_outputs,y_true=all_targets,average='weighted')
        f1score = f1_score(y_pred=all_outputs, y_true=all_targets,average='weighted')
        writer.add_pr_curve('pr_curve', all_targets, all_outputs_probs, global_step=0)
        writer.close()
        
        avg_loss = epoch_loss/total

        if epoch == 0:
            best_model = model
            best_loss = avg_loss
            best_epoch = 0
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = model
                best_epoch = epoch

        

    print(f'Validation Recall Score: {recall}')
    print(f'Validation Precision Score: {precision}')
    print(f'Validation F1 score: {f1score}')

    return best_model, best_loss, best_epoch, f1score, recall, precision, avg_loss


def test(dl, model, batch_size):
    '''
    Testing loop
    Inputs - dl: test data loader
             model: trained model
             batch_size: batch size
    '''

    model.eval()

    epoch_loss = 0.0
    total = 0
    pred_bin_total = 0
    # total_cells = 0
    all_outputs = list()
    all_targets = list()

    with torch.no_grad():
        for X, Y in dl:
            if Y.shape[0] == batch_size:

                pred_scores = model(X)
                pred_scores = pred_scores.view(batch_size,-1)
                wce_loss = bceLoss(pred_scores, Y)

                pred_bin = (pred_scores > config.CLASS_THRESH).float()
                all_outputs.append(pred_bin.view(-1,1).detach().cpu().numpy())
                all_targets.append(Y.view(-1,1).detach().cpu().numpy())
                total += Y.shape[0]
                epoch_loss += wce_loss.item()
                pred_bin_total += pred_bin.sum()
                # total_cells += int(config.TRAIN_BATCH_SIZE*config.LAT_GRIDS*config.LON_GRIDS)
            
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        recall = recall_score(y_pred=all_outputs,y_true=all_targets,average='weighted')
        f1score = f1_score(y_pred=all_outputs, y_true=all_targets,average='weighted')
        precision = precision_score(y_pred=all_outputs,y_true=all_targets,average='weighted')

        report = classification_report(all_targets, all_outputs, output_dict=True)

        avg_loss = epoch_loss/total
        # avg_per_pred_bin = (pred_bin_total/total_cells) * 100

    print(" ")
    print(f'Test Loss: {avg_loss}')
    print(f'Test Recall Score: {recall}')
    print(f'Test Precision Score: {precision}')
    print(f'Test F1 Score: {f1score}')
    # print(f'Average % Predicted Hotspots: {avg_per_pred_bin}')

    return avg_loss, f1score, recall, precision, report

if __name__ == '__main__':
    
    start_epoch = 0

    torch.manual_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    
    prepDatasetsPath = config.PROJECT_DIR + '/Data/PreprocessedDatasets'

    if not os.path.exists(Path(prepDatasetsPath + f'/{config.CRIME_TYPE_NUM}_features.h5')):
        dp = DataPreprocessing(config.PROJECT_DIR)
    
    train_data = DataPreLoader(prepDatasetsPath = prepDatasetsPath,
                                device=device,
                                name = 'train')
    
    val_data = DataPreLoader(prepDatasetsPath = prepDatasetsPath,
                                device=device,
                                name = 'val')

    test_data = DataPreLoader(prepDatasetsPath = prepDatasetsPath,
                                device=device,
                                name = 'test')

    train_loader = DataLoader(train_data, batch_size=config.TRAIN_BATCH_SIZE)
    val_loader = DataLoader (val_data, batch_size=config.TRAIN_BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.TRAIN_BATCH_SIZE)

    model_save_path = config.MODEL_SAVE_PATH
    if not os.path.exists(Path(model_save_path)):
        os.makedirs(Path(model_save_path))

    model = ConvLSTMModel(input_dim=config.CRIME_TYPE_NUM, hidden_dim=config.HIDDEN_DIM, kernel_size=config.KERNEL_SIZE,bias=True)
    optim = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=config.LEARNING_RATE, max_lr=6*config.LEARNING_RATE,cycle_momentum=False,step_size_up=500)
    optim_name = type(optim).__name__

    try:
        checkpoint = torch.load(model_save_path+f'/CheckPoint__bs-({config.TRAIN_BATCH_SIZE})_threshold-({config.CLASS_THRESH})_weights-({config.BCE_WEIGHTS}).pt')
        model.load_state_dict(checkpoint['model'])
        print('\n Model Loaded \n')
        start_epoch = checkpoint['epoch'] + 1
    except:
        pass
    model.to(device)

    try:
        optim.load_state_dict(checkpoint['optim'])
        print('\n Optimizer Loaded \n')
    except:
        pass

    save = True


    print('\n Training Starts \n')

    best_model, best_loss, best_epoch = train(train_dl=train_loader,
                                            val_dl=val_loader,
                                            model=model,
                                            optim=optim,
                                            scheduler=scheduler,
                                            epochs=config.N_EPOCHS,
                                            batch_size=config.TRAIN_BATCH_SIZE,
                                            save=save,
                                            start_epoch=start_epoch,
                                            model_save_path=model_save_path)
                
    test_loss, _ ,test_recall, _ = test(dl=test_loader, model=best_model, batch_size=config.TRAIN_BATCH_SIZE)

    print('\n Saving best model \n')
    print(f'Best model saved at {best_epoch} epoch')

    final_checkpoint = {'model':best_model.state_dict(), 'epoch':best_epoch}
    torch.save(final_checkpoint,model_save_path+f'/BestModel__bs-({config.TRAIN_BATCH_SIZE})_threshold-({config.CLASS_THRESH})_weights-({config.BCE_WEIGHTS}).pt')











