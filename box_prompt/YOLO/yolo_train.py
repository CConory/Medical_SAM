from ultralytics import YOLO
import argparse

model = YOLO('/root/autodl-tmp/Medical_SAM/box_prompt/YOLO/weights/yolov8m.pt')


# model.train(data='bowl_2018.yaml', epochs=300, imgsz=320, batch=16, project="bowl_2018", name="yolov8m_300e_dsbowl")

# model.train(data='bowl_2018.yaml', epochs=300, imgsz=256, batch=16, project="pretrained", name="yolov8m_300e_imgsz256_dsbowl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='bowl_2018.yaml', help='the path of the data yaml')
    parser.add_argument('--epochs', type=int, default=300, help='the training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='the training image size')
    parser.add_argument('--batch', type=int, default=16, help='the training batch size')
    parser.add_argument('--project', type=str, default='pretrained', help='the path of the training weights root directory')
    parser.add_argument('--name', type=str, default='yolov8m_300e_dsbowl', help='the path of the training weights directory')
    args = parser.parse_args()
    
    data_yaml = args.data
    train_epoch = args.epochs
    image_size = args.imgsz
    train_batch = args.batch
    project_dir = args.project
    name_dir = args.name

    model.train(data=data_yaml, epochs=train_epoch, imgsz=image_size, batch=train_batch, project=project_dir, name=name_dir)
    
#     model.train(data=data_yaml, epochs=train_epoch, imgsz=[2040,1536],rect=True, batch=train_batch, project=project_dir, name=name_dir)

    