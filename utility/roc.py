from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from utility.data_loader import *
from model.xception import *

class getModelOutputs:
    def __init__(self, total_csv_dir, test_csv_dir, img_dir, model_dir, save_dir, discriminator, batch_size, img_size):
        self.TOTAL_CSV_DIR = total_csv_dir
        self.TEST_CSV_DIR = test_csv_dir
        self.IMG_DIR = img_dir
        self.MODEL_DIR = model_dir
        self.DISCRIMINATOR = discriminator
        self.BATCH_SIZE = batch_size
        self.IMG_SIZE = img_size
        self.G_MODEL_DIR = os.path.join(self.MODEL_DIR, 'g_model.pth')
        self.L_MODEL_DIR = os.path.join(self.MODEL_DIR, 'l_model.pth')
        self.F_MODEL_DIR = os.path.join(self.MODEL_DIR, 'f_model.pth')
        self.SAVE_DIR = save_dir
        self.HEATMAP_DIR = os.path.join(save_dir, 'heatmaps')
        os.makedirs(self.HEATMAP_DIR, exist_ok=True)
        self.label_save_path = os.path.join(self.MODEL_DIR, 'onehot_labels.csv')
        self.g_save_path = os.path.join(self.MODEL_DIR, 'g_results.csv')
        self.l_save_path = os.path.join(self.MODEL_DIR, 'l_results.csv')
        self.f_save_path = os.path.join(self.MODEL_DIR, 'f_results.csv')
        self.MLB = returnMLB(self.TOTAL_CSV_DIR, self.DISCRIMINATOR).returnMLB()
        self.CLASS_LENGTH = len(self.MLB.classes_)
        self.TEST_DATASET = readTestDataset(self.TEST_CSV_DIR, self.IMG_DIR, self.MLB, self.DISCRIMINATOR, transforms.Compose([testToTensor()]))
        self.FILENAMES = self.TEST_DATASET.df.iloc[:,0].tolist()
        self.DATALOADER = DataLoader(self.TEST_DATASET, self.BATCH_SIZE, shuffle=False, num_workers=0)
        self.loadModels()

    def loadModels(self):
        self.G_MODEL = GlobalNet(self.CLASS_LENGTH).cuda()
        self.L_MODEL = LocalNet(self.CLASS_LENGTH).cuda()
        self.F_MODEL = FusionNet(self.CLASS_LENGTH).cuda()

        self.G_MODEL.load_state_dict(torch.load(self.G_MODEL_DIR))
        self.L_MODEL.load_state_dict(torch.load(self.L_MODEL_DIR))
        self.F_MODEL.load_state_dict(torch.load(self.F_MODEL_DIR))
        self.G_MODEL.eval()
        self.L_MODEL.eval()
        self.F_MODEL.eval()

    def importData(self,data):
        inputs, labels = data['image'], data['label']
        inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda()
        return inputs, labels

    def concatenateOutputs(self,nparray, outputs):
        outputs = outputs.detach().cpu().numpy()
        nparray = np.concatenate((nparray, outputs), axis=0)
        return nparray

    def numpyToDataframe(self, nparray, columns):
        df = pd.DataFrame(data=nparray, columns=columns)
        FILENAMES = pd.DataFrame(data=self.FILENAMES, columns=['FILENAME'])
        df = pd.concat((FILENAMES, df), axis=1)
        return df

    def getOutputs(self):
        print('Getting outputs from the models')
        g_results = np.array([[]])
        l_results = np.array([[]])
        f_results = np.array([[]])
        onehot_labels = np.array([[]])
        for i, data in enumerate(tqdm(self.DATALOADER)):
            inputs, labels = self.importData(data)
            g_outputs, g_poolings, heatmaps = self.G_MODEL.gOutput(inputs)
            self.L_MODEL.forward(heatmaps)
            l_outputs, l_poolings = self.L_MODEL.forward(heatmaps)
            concated_poolings = torch.cat((g_poolings, l_poolings), dim=1)
            f_outputs = self.F_MODEL.forward(concated_poolings)
            if i == 0:
                onehot_labels = labels.detach().cpu().numpy()
                g_results = g_outputs.detach().cpu().numpy()
                l_results = l_outputs.detach().cpu().numpy()
                f_results = f_outputs.detach().cpu().numpy()
                self.saveHeatMap(heatmaps, labels)
            else:
                onehot_labels = self.concatenateOutputs(onehot_labels, labels)
                g_results = self.concatenateOutputs(g_results, g_outputs)
                l_results = self.concatenateOutputs(l_results, l_outputs)
                f_results = self.concatenateOutputs(f_results, f_outputs)

        self.onehot_label_df = self.numpyToDataframe(onehot_labels, self.MLB.classes_)
        self.g_df = self.numpyToDataframe(g_results, self.MLB.classes_)
        self.l_df = self.numpyToDataframe(l_results, self.MLB.classes_)
        self.f_df = self.numpyToDataframe(f_results, self.MLB.classes_)

        print('Saving outputs')
        self.onehot_label_df.to_csv(self.label_save_path, index=False)
        self.g_df.to_csv(self.g_save_path, index=False)
        self.l_df.to_csv(self.l_save_path, index=False)
        self.f_df.to_csv(self.f_save_path, index=False)

    def saveHeatMap(self, heatmaps, labels):
        heatmaps = heatmaps.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        j = 0
        for heatmap, label in zip(heatmaps, labels):
            heatmap = np.transpose(heatmap, (1,2,0))
            heatmap = heatmap * 255
            heatmap = heatmap.astype(np.uint8)
            label = [True if l == 1. else False for l in label]
            filename = f'{j}_{"_".join(self.MLB.classes_[label])}.jpg'
            save_path = os.path.join(self.HEATMAP_DIR, filename)
            print(save_path)
            cv2.imwrite(save_path, heatmap)
            j += 1

    def saveHeatMaps(self):
        for i, data in enumerate(tqdm(self.DATALOADER)):
            inputs, labels = self.importData(data)
            g_outputs, g_poolings, heatmaps = self.G_MODEL.gOutput(inputs)
            self.saveHeatMap(heatmaps, labels)

class Evaluation:
    def __init__(self, onehot_df, g_df, l_df, f_df,save_dir):
        self.ONEHOT_DF = onehot_df
        self.G_DF = g_df
        self.L_DF = l_df
        self.F_DF = f_df
        self.CLASSES = self.G_DF.columns.tolist()[1:]
        self.SAVE_DIR = save_dir
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    def evaluate(self, label):
        y = np.array(self.ONEHOT_DF[label].tolist())
        g_prob = np.array(self.G_DF[label].tolist())
        l_prob = np.array(self.L_DF[label].tolist())
        f_prob = np.array(self.F_DF[label].tolist())

        g_fpr, g_tpr, _ = roc_curve(y, g_prob)
        l_fpr, l_tpr, _ = roc_curve(y, l_prob)
        f_fpr, f_tpr, _ = roc_curve(y, f_prob)
        g_auc = auc(g_fpr, g_tpr)
        l_auc = auc(l_fpr, l_tpr)
        f_auc = auc(f_fpr, f_tpr)

        return g_fpr, g_tpr, g_auc, l_fpr, l_tpr, l_auc, f_fpr, f_tpr, f_auc

    def saveROCPlot(self, label, g_fpr, g_tpr, g_auc, l_fpr, l_tpr, l_auc, f_fpr, f_tpr, f_auc):
        plt.title('Receiver Operating Characteristic(ROC)')
        plt.xlabel('False Positive Rate(1 - Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')

        plt.plot(g_fpr, g_tpr, 'r', label='Global Branch (AUC = %0.5f)' % g_auc)
        plt.plot(l_fpr, l_tpr, 'g', label='Local Branch (AUC = %0.5f)' % l_auc)
        plt.plot(f_fpr, f_tpr, 'b', label='Fusion Branch (AUC = %0.5f)' % f_auc)

        plt.legend(loc='lower right')
        save_path = os.path.join(self.SAVE_DIR, f'ROC_{label}.png')
        plt.savefig(save_path)
        print(save_path, 'is saved!')
        plt.cla()
        plt.clf()
        plt.close()

    def makeDataFrame(self, auc_list):
        auc_df = pd.DataFrame(data=auc_list,columns=['Global_branch_AUC','Local_branch_AUC','Fusion_branch_AUC'])
        auc_df.index = self.CLASSES
        auc_df.loc['Mean'] = auc_df.mean()
        auc_df.index.name = 'Class'
        auc_df.reset_index(level=['Class'], inplace=True)
        return auc_df

    def execute(self):
        g_auc_list = []
        l_auc_list = []
        f_auc_list = []
        for label in self.CLASSES:
            g_fpr, g_tpr, g_auc, l_fpr, l_tpr, l_auc, f_fpr, f_tpr, f_auc = self.evaluate(label)
            g_auc_list.append(g_auc)
            l_auc_list.append(l_auc)
            f_auc_list.append(f_auc)
            self.saveROCPlot(label, g_fpr, g_tpr, g_auc, l_fpr, l_tpr, l_auc, f_fpr, f_tpr, f_auc)
        g_auc_list, l_acu_list, f_auc_list = np.array(g_auc_list), np.array(l_auc_list), np.array(f_auc_list)
        total_auc = np.vstack((g_auc_list, l_auc_list, f_auc_list)).T
        total_df = self.makeDataFrame(total_auc)
        save_path = os.path.join(self.SAVE_DIR, 'auc.csv')
        total_df.to_csv(save_path, index=False)
