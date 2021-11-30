from utility.ops import *
from utility.data_loader import *
from utility.xception_train_validation import *
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

CSV_PATH = 'D:/DATA/viennacode_img_19600101_20191231_unique_preprocessed/labels_middle.csv'
IMG_DIR = 'D:/DATA/viennacode_img_19600101_20191231_unique_preprocessed/imgs'
SAVE_NAME = 'XCEPTION_LR_0001_EPOCH_100_20210810'
TOTAL_EPOCH = 100
BATCH_SIZE = 16
TRAIN_RATIO = 0.7
LR = 0.001
DISCRIMINATOR = '|'

device = torch.device("cuda")

dataloader = returnDataLoader(csv_path=CSV_PATH,
                              train_ratio=TRAIN_RATIO,
                              img_dir=IMG_DIR,
                              discriminator=DISCRIMINATOR)

train_ = returnLossAndAcc(dataset=dataloader.TRAIN_DATASET,
                          batch_size=BATCH_SIZE)
val_ = returnLossAndAcc(dataset=dataloader.VAL_DATASET,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

save = Save(save_name=SAVE_NAME,
            total_epoch=TOTAL_EPOCH,
            train_=train_,
            val_=val_)

train_and_val = TrainAndValidation(save=save,
                                   train_=train_,
                                   val_=val_,
                                   class_length=len(dataloader.MLB.classes_),
                                   learning_rate=LR,
                                   total_epoch=TOTAL_EPOCH)
train_and_val.execute()