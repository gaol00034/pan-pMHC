import logging
from sklearn.metrics import *
import torch

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def generatescore(model, tuple, device):
    preds = []
    for ind, data in enumerate(tuple):
        pepseqEmb = (data['pep']).to(device)
        ta = (data['tcr']).to(device)
        taadj = data['tcradj'].to(device)
        ta_n_list = data['tcr_n_list'].to(device)
        scores = model(pepseqEmb, ta, taadj, ta_n_list)
        preds.extend(scores)
    return preds

def predict(model, tuple, device):
    model = model.eval()
    model = model.to(device)
    predlist = generatescore(model, tuple, device)
    return predlist
def test(model, testtuple, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    preds = predict(model, testtuple, device)
    tauc = 0
    taupr = 0
    for ind in range(len(preds)):
        pred = preds[ind]
        label = testtuple['label'][ind]
        val_auc = roc_auc_score(torch.tensor(label), torch.tensor(pred))
        precision, recall, _ = precision_recall_curve(torch.tensor(label), torch.tensor(pred))
        val_aupr = auc(recall, precision)
        tauc += val_auc
        taupr += val_aupr

    N = len(preds)
    aaupr = taupr / N
    aauc = tauc / N
    return aauc, aaupr

def train_epoch(model, tuple, lr, weight_decay, device):
    trainloss=0
    for ind, data in enumerate(tuple):
        pepseqEmb = (data['pep']).to(device)
        label = (data['label']).to(device)
        ta = (data['tcr']).to(device)
        taadj = data['tcradj'].to(device)
        ta_n_list = data['tcr_n_list'].to(device)
        scores = model(pepseqEmb, ta, taadj, ta_n_list)
        loss_func = nn.BCELoss()
        loss = loss_func(scores.squeeze(), label)
        trainloss = trainloss + loss.item()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        opt.zero_grad()
        loss.backward()


def fit(model, traintuple, lr, epoch, weight_decay, device, testtuple):
    model = model.to(device)
    bestauc = 0
    earlystop = 0
    for i in range(epoch):
        train_epoch(model, traintuple, lr, weight_decay, device)
        auc, aupr = test(model, testtuple)
        if auc>bestauc:
            model_save_path = os.path.join(modelname)
            torch.save(model.state_dict(), model_save_path)
            bestauc = auc
            earlystop = 0
        else:
            earlystop += 1

        logger.info('Epoch:[{}/{}]\t trainloss={:.9f}\t testauc={:.9f}\t testaupr={:.9f}'.format(i + 1, epoch, trainloss / len(ids), auc, aupr))
        if earlystop == 10:
            logger.info(
                'finish')
            break
