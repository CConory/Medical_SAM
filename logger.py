import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from evaluate_from_pt import show_box
import wandb
import numpy as np
from evaluate_from_pt import show_mask

def tag_images(img,objs,color="green",category_dict=None):
    '''
    objs = [N,5] for targets [ xyxy, category_id]
         = [N,6] for predn, [xyxy,score,category_id]
    '''
    W,H = img.size
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 10)
    for i,obj in enumerate(objs):
        box = obj[:4]
        draw.rectangle(box,outline=color,width=4)
        x1,y1,x2,y2 = box
        if len(obj) == 6:
            label = label = category_dict[int(obj[-1])] if category_dict else str(int(int(obj[-1])))
            label +=  '({:.2f})'.format(obj[4])
        else:
            label = category_dict[int(obj[-1])] if category_dict else str(int(int(obj[-1])))
        if len(obj) > 4:
            w,h = font.getsize(label)
            if x1+w > W or y2+h > H:
                draw.rectangle((x1, y2-h, x1 + w, y2), fill=color)
                draw.text((x1,y2-h),label,fill='white',font=font)
            else:
                draw.rectangle((x1, y2, x1 + w, y2 + h), fill=color)
                draw.text((x1,y2),label,fill='white',font=font)
    return img1

def visualization_bboxes(image,target,predn,category_dict = None,img_style="PIL"):
    if img_style == "Numpy":
        image = Image.fromarray(image)
    image = tag_images(image,target,color = "green",category_dict = category_dict)
    image = tag_images(image,predn,color = "blue",category_dict = category_dict)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    # for box in target: 
    #     show_box(box, plt.gca(),"yellow")
    # for box in predn: 
    #     show_box(box, plt.gca(),"#0099FF")
    plt.axis('off')
    return plt
    # return wandb.Image(plt)

def visualization_masks(image,target_mask,pred_mask=None,img_style="plt"):
    if pred_mask is None:
        pred_mask = np.zeros_like(target_mask).astype(np.bool)
    target_mask = target_mask.astype(np.bool)
    pred_mask = pred_mask.astype(np.bool)
    inter_mask = (pred_mask) & (target_mask)
    show_mask(pred_mask, image.gca(),np.array([0/255, 0/255, 255/255, 0.4]))
    show_mask(target_mask, image.gca(),np.array([0/255, 255/255, 0/255, 0.4]))
    show_mask(inter_mask, image.gca(),np.array([255/255, 0, 0/255, 0.4]))
    return image

def wandb_logger(result,visulization_imgs,args,category_names):
    class_id2name = {id:name for id,name in enumerate(category_names)}
    f1 = result['f1']
    p = result['precision']
    r = result['recall']
    conf = f1.mean(0).argmax()

    print("### Evaluation Result ###")
    print(" The best Mean F1 score is under conf: ".ljust(5, ' '),round(conf / 1000, 2))
    print(" ")
    print("Class_Name".ljust(20, ' '), f"AP_{int(args.mAP_threshold*100)}".ljust(15, ' '),"Recall".ljust(15, ' '),"Precision".ljust(15, ' '),"F1".ljust(15, ' '))
    print(" ")

    data = []
    for i,class_id in enumerate(result["classes"]):
        class_name = class_id2name[class_id]
        print(f'{class_name}'.ljust(20, ' '), f'{result["ap"][i]*100:.2f}'.ljust(15, ' '), f'{r[i][conf]*100:.2f}'.ljust(15, ' '),f'{p[i][conf]*100:.2f}'.ljust(15, ' '),f'{f1[i][conf]*100:.2f}'.ljust(15, ' '))
        data.append([class_name,result["ap"][i]*100,r[i][conf]*100,p[i][conf]*100,f1[i][conf]*100,round(conf / 1000, 2)])
    print('Mean'.ljust(20, ' '), f'{result["ap"].mean(0)*100:.2f}'.ljust(15, ' '), f'{r.mean(0)[conf]*100:.2f}'.ljust(15, ' '),f'{p.mean(0)[conf]*100:.2f}'.ljust(15, ' '),f'{f1.mean(0)[conf]*100:.2f}'.ljust(15, ' '))
    data.append(["Mean",result["ap"].mean(0)*100,r.mean(0)[conf]*100,p.mean(0)[conf]*100,f1.mean(0)[conf]*100,round(conf / 1000, 2)])
    table = wandb.Table(data=data, columns = ["Class_Name", f"AP_{int(args.mAP_threshold*100)}","Recall","Precision","F1"," Best Confidence"])
    wandb.log({"val/result": table})

    wandb.log({
            f"test/mAP_{int(args.mAP_threshold*100)}": result["ap"].mean(0)*100,
            "test/recall": r.mean(0)[conf]*100,
            "test/precsion":p.mean(0)[conf]*100,
            "test/f1":f1.mean(0)[conf]*100,
            })
    
    # 类别太多了，画的太慢了
    # for index,class_id in enumerate(result["classes"][:5]):
    #     class_name = category_names[class_id]
    #     recall = result['recall'][index]
    #     precision = result['precision'][index]
    #     data = [[x, y] for (x, y) in zip(recall, precision)]
    #     table = wandb.Table(data=data, columns = ["recall", "precision"])
    #     plot = wandb.plot.line(table, "recall", "precision", stroke=None, title="Average Precision: "+class_name)
    #     wandb.log({"val/AP/"+class_name : plot})
    
    class_name = "Mean"
    recall = result['recall'].mean(0)
    precision = result['precision'].mean(0)
    data = [[x, y] for (x, y) in zip(recall, precision)]
    table = wandb.Table(data=data, columns = ["recall", "precision"])
    plot = wandb.plot.line(table, "recall", "precision", stroke=None, title="Average Precision: "+class_name)
    wandb.log({"val/AP/"+class_name : plot})

    wandb.log({"visualization": visulization_imgs})