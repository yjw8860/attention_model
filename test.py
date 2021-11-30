from utility.roc import *
import re

TOTAL_CSV_DIR = 'D:/data/viennacode_img_19600101_20191231_unique_preprocessed/labels_middle.csv'
TEST_CSV_DIR = re.sub('labels_middle', 'test_labels_middle', TOTAL_CSV_DIR)
IMG_DIR = re.sub('labels_middle.csv', 'imgs', TOTAL_CSV_DIR)
MODEL_DIR = './XCEPTION_LR_0001_EPOCH_100_20210810'
SAVE_DIR = os.path.join(MODEL_DIR, 'result')
DISCRIMINATOR = "|"
BATCH_SIZE = 8
IMG_SIZE = 224
os.makedirs(SAVE_DIR, exist_ok=True)

getmodeloutput = getModelOutputs(TOTAL_CSV_DIR, TEST_CSV_DIR, IMG_DIR, MODEL_DIR, SAVE_DIR, DISCRIMINATOR, BATCH_SIZE, IMG_SIZE)
getmodeloutput.getOutputs()
getmodeloutput.saveHeatMaps()