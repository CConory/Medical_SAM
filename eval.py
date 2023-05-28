import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os 
import cv2
from tqdm import tqdm


if __name__ == '__main__':

    df = pd.read_csv('segpc2021/test_train_data.csv')
    imgs_test = df[df.category=='test']['image_id']
    imgs_val = df[df.category=='validation']['image_id']
    test_files = list(imgs_test)
   
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    dice_score = []
    
    for file_name in tqdm(test_files):

        img_id = list(file_name.split('.'))[0]

        pred_path = f'segpc2021/point_0/mask/mix/{img_id}.png'
        pred_img = cv2.imread(pred_path, 0)

        gts = os.listdir(f'segpc2021/data/images/y/{img_id}/')

        gt_zeros = np.zeros(pred_img.shape)

        for gt in gts:

            gt_path = f'segpc2021/data/images/y/{img_id}/{gt}'
            gt_img = cv2.imread(gt_path, 0)
            gt_zeros += gt_img
            
        cfsmat = confusion_matrix(gt_zeros.flatten(), pred_img.flatten())
        
        sum_iou = 0
        sum_prec = 0
        sum_acc = 0
        sum_recall = 0
        sum_dice = 0
        
        for i in range(3):
            tp = cfsmat[i,i]
            fp = np.sum(cfsmat[0:3,i]) - tp
            fn = np.sum(cfsmat[i,0:3]) - tp
            
            
            tmp_iou = tp / (fp + fn + tp)
            tmp_prec = tp / (fp + tp + 1) 
            tmp_acc = tp
            tmp_recall = tp / (tp + fn)
            
            
            sum_iou += tmp_iou
            sum_prec += tmp_prec
            sum_acc += tmp_acc
            sum_recall += tmp_recall
            
        
        sum_acc /= (np.sum(cfsmat)) 
        sum_prec /= 3
        sum_recall /= 3
        sum_iou /= 3
        sum_dice = 2 * sum_prec * sum_recall / (sum_prec + sum_recall)
        
        iou_score.append(sum_iou)
        acc_score.append(sum_acc)
        pre_score.append(sum_prec)
        recall_score.append(sum_recall)
        dice_score.append(sum_dice)

        
print('mean IoU: {:.4f}, {:.4f}'.format(np.mean(iou_score),np.std(iou_score)))
print('mean accuracy: {:.4f}, {:.4f}'.format(np.mean(acc_score),np.std(acc_score)))
print('mean precsion: {:.4f}, {:.4f}'.format(np.mean(pre_score),np.std(pre_score)))
print('mean recall: {:.4f}, {:.4f}'.format(np.mean(recall_score),np.std(recall_score)))
print('mean Dice: {:.4f}, {:.4f}'.format(np.mean(dice_score),np.std(dice_score)))