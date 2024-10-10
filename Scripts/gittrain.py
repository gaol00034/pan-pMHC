from dataset import *
from train import *
from network.sgtpmi import SGTPMI
from torch import *
import copy

def load_checkpoint(filepath, pepmodel):
    state_dict = copy.deepcopy(pepmodel.state_dict())
    params = torch.load(filepath, map_location=torch.device('cpu'))
    print()
    for key in params:
        if key in state_dict:
            state_dict[key] = params[key]
    return state_dict


def main():
    logging.basicConfig(
        filename='logs/train.log',
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    train_auc_result = []
    train_aupr_result = []
    train_loss_result = []
    test_auc_result = []
    test_aupr_result = []
    test_loss_result = []
    for fold in [0,1,2,3,4]:
        logging.info(f"----------------fold{fold} start.----------------")
        train_data_path = f"./fold{fold}/train.csv"
        test_data_path = f"./fold{fold}/test.csv"

        embed_output_dir = f"./fold{fold}/"
        if not os.path.exists(embed_output_dir):
            os.makedirs(embed_output_dir)

        train_loader = DataLoader(
            read_data(train_data_path),
            task,
            batch_size,
            features,
            tcr_padding_length,
            peptide_padding_length,
        )

        test_loader = DataLoader(
            read_data(test_data_path),
            task,
            batch_size,
            features,
            tcr_padding_length,
            peptide_padding_length,
        )


        logging.info(
            "Dataset: {}, Task2 train set num: {}, Task2 test set num: {}".format(
                dataset, len(train_loader), len(test_loader)
            )
        )

        model = SGTPMI(
            tcr_padding_len=tcr_padding_length,
            peptide_padding_len=peptide_padding_length,
            map_num=len(features),
            dropout_prob=dropout,
            hidden_channel=hidden_channel,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=param["lr"])

        criterion = nn.BCELoss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            logger=logging.getLogger(__name__),
            save_dir=embed_output_dir,
            patience=10
        )

        fold_train_loss, fold_test_loss, fold_train_auc, fold_test_auc, fold_train_aupr, fold_test_aupr = trainer.train(
            train_loader=train_loader, test_loader=test_loader, epochs=epoch
        )

        train_auc_result.append(fold_train_auc)
        train_aupr_result.append(fold_train_aupr)
        train_loss_result.append(fold_train_loss)
        test_auc_result.append(fold_test_auc)
        test_aupr_result.append(fold_test_aupr)
        test_loss_result.append(fold_test_loss)

    logging.info(
        "\nfinish!!\n train loss: %.4f±%.4f, test loss: %.4f±%.4f \n train auc: %.4f±%.4f, test auc: %.4f±%.4f \n train aupr: %.4f±%.4f, test aupr: %.4f±%.4f" % (
            np.mean(train_loss_result), np.std(train_loss_result), np.mean(test_loss_result), np.std(test_loss_result),
            np.mean(train_auc_result), np.std(train_auc_result), np.mean(test_auc_result), np.std(test_auc_result),
            np.mean(train_aupr_result), np.std(train_aupr_result), np.mean(test_aupr_result), np.std(test_aupr_result),
        )
    )

if __name__ == "__main__":
    main()
