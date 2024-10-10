from dataset import *
from abtrain import *
from network.sgtpmi import SGTPMI
from torch import *
import copy
from sklearn.metrics import recall_score

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
        filename='logs/predict.log',
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    for fold in [0,1,2,3,4]:
        logging.info(f"----------------fold{fold} start.----------------")
        test_data_path = f"./predict.csv"
        embed_output_dir = f"ckpt/TITAN/{args.split_mode}/fold{fold}/"
        if not os.path.exists(embed_output_dir):
            os.makedirs(embed_output_dir)


        test_loader = DataLoader(
            read_data(test_data_path),
            task,
            batch_size,
            features,
            tcr_padding_length,
            peptide_padding_length
        )


        logging.info(
            "Dataset: {}, Task2 test set num: {}".format(
                dataset, len(test_loader)
            )
        )

        model = SGTPMI(
            tcr_padding_len=tcr_padding_length,
            peptide_padding_len=peptide_padding_length,
            map_num=len(features),
            dropout_prob=dropout,
            hidden_channel=hidden_channel,
        )

        ps = load_checkpoint("./model.pt", model)
        model.load_state_dict(ps)
        model.to(device)

        model.eval()

        with torch.no_grad():
            val_targets = []
            val_preds = []

            for data in test_loader:
                torch.cuda.empty_cache()
                output = model(data)
                val_preds.extend(output)
                val_targets.extend(data["labels"])
            val_auc = roc_auc_score(torch.tensor(val_targets), torch.tensor(val_preds))
            precision, recall, _ = precision_recall_curve(torch.tensor(val_targets), torch.tensor(val_preds))
            val_aupr = auc(recall, precision)
            recall = recall_score(torch.tensor(val_targets), torch.tensor(val_preds)>threshold)
            logging.info(
                "test_auc: {:.6f}, test_recall:{:.6f}, test_aupr : {:.6f}".format(
                    val_auc, recall, val_aupr
                )
            )


if __name__ == "__main__":
    main()
