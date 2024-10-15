import torch
import cv2
import numpy as np

# diretorios copiados das pastas o yolo (D:\bkp note antigo\JKU\testeYolo\bikeParts\yolov5)
# copiei ainda o arquivo export.py do mesmo diretorio para a pasta do projeto (C:\Users\Pedro\PycharmProjects\niryoProject)
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords


def readModel(ptFile):
    model = DetectMultiBackend(ptFile)
    names = model.names
    # model = torch.hub.load('ultralytics/yolov5', 'custom', ptFile)  # custom trained model
    # names = model.names
    return model, names


def prepareImage(img, model):
    imgs = [None]
    imgs[0] = img
    im0 = imgs.copy()
    im = np.stack([letterbox(x, new_shape=(640, 640), stride=64, auto=True)[0] for x in im0])  # resize
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous

    im = torch.from_numpy(im).to(torch.device('cpu'))
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


def inference(model, img):
    im = prepareImage(img, model)
    results = model(im)
    results = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
    return results, im


def selectObjects(pred, im, im0, names, thres_conf=0.25, draw=True):
    objList = []
    newImg = im0.copy()
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                id = int(cls)  # integer class
                label = f'{names[id]} {conf:.2f}'
                if conf >= thres_conf:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    if draw:
                        cv2.rectangle(newImg, (x1, y1), (x2, y2), (90, 0, 255), 2)
                        cv2.putText(newImg, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 0), 1)
                    objList.append([id, float(conf), x1, y1, x2, y2])
    return objList, newImg


#
# def readModel(ptFile):
#   model = torch.hub.load('ultralytics/yolov5', 'custom', ptFile)  # custom trained model
#   names = model.names
#   return model, names
#
# def inference(model, im):
#   # Inference
#   results = model(im)
#   # results = non_max_suppression(results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#   return results
#
# def selectObjects(results, img, names, thres_conf = 0.25):
#     objList = []
#     newImg = img.copy()
#     if len(results):
#         # results.show()
#         for i, obj in enumerate(results.xyxy[0]):
#             # print(obj)
#             id   = int(obj[5])
#             conf = float(obj[4])
#             # print(conf, thres_conf)
#             if conf >= thres_conf:
#                 x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
#                 cv2.rectangle(newImg, (x1, y1), (x2, y2), (90, 0, 255), 2)
#                 cv2.putText(newImg, names[id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                             (0, 0, 0), 2)
#                 # roi = img[y1:y2, x1:x2]
#                 # cv2.imshow(f"roi-{i}", roi)
#                 # print(f"{id} {names[id]} {conf} {x1}, {y1}, {x2}, {y2} ")
#                 objList.append([id, conf, x1, y1, x2, y2])
#     return objList, newImg

def centerObjects(objList, idSearch):
    count, cx, cy = 0, 0, 0
    if len(objList):
        for i, obj in enumerate(objList):
            id, conf, x1, y1, x2, y2 = objList[i]
            # found the center of n objects
            # if id == idSearch[0]:
            #     cx += x1 + (x2-x1)/2
            #     cy += y1 + (y2-y1)/2
            #     count += 1

            # found the center of the first object
            if id == idSearch[0]:
                # cx = x1 + (x2-x1)/2
                # cy = y1 + (y2-y1)/2
                cx = x1 + (x2 - x1)
                cy = y1 + (y2 - y1) / 2
                count += 1

    # if count > 0:
    #     # found the center of n objects
    #     # cx = cx / count
    #     # cy = cy / count
    #     pass
    # else:
    #     print(f"{idSearch[1]} not found")

    return cx, cy, count


def main():
    model, names = readModel('D:/bkp note antigo/JKU/testeYolo/yolov5-master/best.pt')
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        # imName = 'D:/bkp note antigo/JKU/testeYolo/yolov5-master/screws/test/washer.7ffd8c02-116f-11ed-8cd3-6432a89e21d0.jpg'  # or file, Path, URL, PIL, OpenCV, numpy, list
        # img = cv2.imread(imName)

        results = inference(model, img)

        # results = non_max_suppression(results, conf_thres=0.5)

        # Results
        results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
        # results.xyxy[0]  # im predictions (tensor)
        # results.pandas().xyxy[0]  # im predictions (pandas)
        # print(results.pandas().xyxy[0])  # im predictions (pandas)
        #      xmin    ymin    xmax   ymax  confidence  class    name
        # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
        # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
        # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
        objList, newImg = selectObjects(results, img, names, thres_conf=0.25)
        print(objList)
        cv2.imshow('Output', newImg)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

