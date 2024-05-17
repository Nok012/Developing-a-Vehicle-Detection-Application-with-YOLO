import math
import sys
import numpy as np
from PyQt5.QtCore import QEvent,Qt
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QInputDialog
from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
import copy
from sort import *
import cv2, imutils
from ultralytics import YOLO


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('ui/ui.ui',self)
        self.setWindowTitle("Car Counter App")

        self.loadBtn.clicked.connect(self.loadVideo)
        self.startBtn.clicked.connect(self.playVideoBtn)
        self.detecBtn.clicked.connect(self.detecVideoBtn)
        self.createBtn.clicked.connect(self.createAreaBtn)
        
        self.startBtn.setEnabled(False)
        self.detecBtn.setEnabled(False)
        self.createBtn.setEnabled(False)

        self.label.installEventFilter(self)
        
        self.run = False
        self.area = []
        self.areaName = []
        self.totalArea = []
        self.pointText = []
        self.numPoint = 0
        self.sizeTatalCars = [set(),set(),set(),set(),set()]
        self.totalCars = []
        
        self.model = YOLO("../Yolo/yolov8l.pt")
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                    "teddy bear", "hair drier", "toothbrush"
                    ]
        self.tracker = None
  
    def openInputDialog(self):
        name, ok = QInputDialog.getText(self, 'Name', 'Area Name')
        if ok:
            self.areaName.append(name)
        print( self.areaName)

    def createAreaBtn(self):
        if not self.create:
            self.createBtn.setText('Not Create')      
        else:
            self.createBtn.setText('Create')
        self.create = not self.create
       
    def playVideoBtn(self):   
        if not self.start:
            self.startBtn.setText('Stop')      
        else:
            self.startBtn.setText('Start')
        self.start = not self.start

    def detecVideoBtn(self):
        if not self.detec:
            self.detecBtn.setText('Not Track')      
        else:
            self.detecBtn.setText('Track')
        self.detec = not self.detec
       
    def closeEvent(self, _):
        self.run = not self.run
    
       
    def eventFilter(self, obj, event):
    
        if obj == self.label and event.type() == QEvent.MouseButtonPress :

            if self.create:

                if event.button() == Qt.LeftButton and self.numPoint == 5:
                    self.totalArea.append([self.areaName[len(self.areaName)-1],self.pointText[0],self.area])
                    self.totalCars.append(self.sizeTatalCars)
                    self.sizeTatalCars = [set(),set(),set(),set(),set()]
                    
                    self.numPoint = 0
                    self.area = []
                    self.pointText = []

                else:
                    if event.button() == Qt.LeftButton and self.numPoint == 4:
                        self.pointText.append((event.pos().x(), event.pos().y()))
                        self.numPoint+= 1
                        self.openInputDialog()

                    if event.button() == Qt.LeftButton and self.numPoint < 4:                   
                        self.area.append((event.pos().x(), event.pos().y()))
                        self.numPoint+= 1

                    if event.button() == Qt.RightButton and self.numPoint == 5:
                        self.pointText.pop()
                        self.areaName.pop()
                        self.numPoint-= 1

                    elif event.button() == Qt.RightButton and self.numPoint > 0:
                        self.area.pop()
                        self.numPoint-= 1
                                                  
                print("Area: ",self.area)
                print("PointText: ",self.pointText)
                print("TotalArea: ",self.totalArea)
                print("Name: ",self.areaName)

        return super().eventFilter(obj, event)

    def loadVideo(self):
        
        self.tracker = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
        dialog = QFileDialog(self)
        dialog.setDirectory(r'C:\Users\dlnno\Desktop\Project\Video')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("Video (*.mp4)")
        dialog.setViewMode(QFileDialog.ViewMode.List)

        if dialog.exec():
            
            filenames = dialog.selectedFiles()
        try:
            cap = cv2.VideoCapture(str(filenames[0]))
            self.run = True
            self.stop = False
            self.start = False
            self.detec = False
            self.create = False
            
            self.detecBtn.setText('Track')
            self.startBtn.setText('Start')
            self.createBtn.setText('Create')
            # self.label.clear()

        except UnboundLocalError:
            return print("Not select")
        
        while self.run:

            QApplication.processEvents()

            if self.stop and self.start == False:

                self.createBtn.setEnabled(True)
                self.startBtn.setEnabled(True)
                self.detecBtn.setEnabled(True)
                
                if self.detec:
                    if self.create:
                        self.cacheImage_2 = copy.deepcopy(self.image)
                        self.polylinesCar(self.cacheImage_2,self.area)
                        self.setPhoto(self.cacheImage_2)
                    else:
                        self.setPhoto(self.image)
                else:
                    if self.create:
                        self.cacheImage_2 = copy.deepcopy(self.cacheImage_1)
                        self.polylinesCar(self.cacheImage_2,self.area)
                        self.setPhoto(self.cacheImage_2)
                    else:
                        self.setPhoto(self.cacheImage_1)
            else:
                success, self.image = cap.read()
                
                self.loadBtn.setEnabled(False)
                self.createBtn.setEnabled(False)
                if success:
                    self.image  = imutils.resize(self.image ,height = 720,width=1280 )
                    self.cacheImage_1 = copy.deepcopy(self.image)
                    results = self.model(self.image, stream=True, verbose=False)
                    self.detecCar(results,self.image)
 
                    if self.detec:                    
                        if self.create:
                            self.cacheImage_2 = copy.deepcopy(self.image)
                            self.polylinesCar(self.cacheImage_2,self.area)
                            self.setPhoto(self.cacheImage_2)
                        else:
                            self.setPhoto(self.image)
                    else:
                        if self.create:
                            self.cacheImage_2 = copy.deepcopy(self.cacheImage_1)
                            self.polylinesCar(self.cacheImage_2,self.area)
                            self.setPhoto(self.cacheImage_2)
                        else:
                            self.setPhoto(self.cacheImage_1)
                else:
                    print("End Video")
                    self.loadBtn.setEnabled(True)
                    self.startBtn.setText('Start')
                    self.detecBtn.setText('Track')
                    self.createBtn.setText('Create')
                    self.startBtn.setEnabled(False)
                    self.detecBtn.setEnabled(False)
                    self.createBtn.setEnabled(False)
                    self.stop = False
                    self.start = False
                    self.detec = False
                    self.run = False

                    self.area = []
                    self.totalArea = []
                    self.pointText = []
                    self.numPoint = 0
                    self.sizeTatalCars = [set(),set(),set(),set(),set()]
                    self.totalCars = []
                    self.areaName = []
                    break
                 
            if self.stop == False:
                self.stop = not self.stop
        
        cap.release()

    def setPhoto(self,image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))
        self.pixmap_image = QPixmap.fromImage(image)

    def detecCar(self,results,image):
        detections_1 = np.empty((0,5))
        detections_2 = np.empty((0,6))


        for r in results:
            boxes = r.boxes   
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1                   
                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0]*100) / 100

                if ((self.classNames[cls] == "car" or self.classNames[cls] == "bus" or self.classNames[cls] == "truck") and conf > 0.6) or self.classNames[cls] == "motorbike":
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections_2 = np.concatenate((detections_2, [(x1,y1,x2,y2,conf,cls)]), axis=0)
                    detections_1 = np.vstack((detections_1, currentArray))
       
        resultsTracker = self.tracker.update(detections_1)

        for r in resultsTracker:
            x1,y1,x2,y2,Id = r
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x1 + w) // 2
            cy = (y1 + y1 + h) // 2

            # x1,y1,w,h,Id = r
            # x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            # cx = (x1 + x1 + w) // 2
            # cy = (y1 + y1 + h) // 2
            
            for r2 in detections_2:
                x1,y1,x2,y2,conf,cls2 = r2
                cls = int(cls2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx2 = (x1 + x1 + w) // 2
                cy2 = (y1 + y1 + h) // 2
                dist = math.hypot(abs(cx-cx2),abs(cy-cy2))

                if dist < 20:
                    if self.classNames[cls] == "car":
                        cv2.circle(image, (cx,cy),2,(0,0,255),-1)
                        cv2.putText(image, f'Id{int(Id)} { self.classNames[cls]} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,0,255), 1)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
                    elif  self.classNames[cls] == "truck":
                        cv2.circle(image, (cx,cy),2,(0,255,0),-1)
                        cv2.putText(image, f'Id{int(Id)} { self.classNames[cls]} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,255,0), 1)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                    elif  self.classNames[cls] == "bus":
                        cv2.circle(image, (cx,cy),2,(255, 0, 255),-1)
                        cv2.putText(image, f'Id{int(Id)} { self.classNames[cls]} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 255), 1)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    elif  self.classNames[cls] == "motorbike":
                        cv2.circle(image, (cx,cy),2,(0,232,255),-1)
                        cv2.putText(image, f'Id{int(Id)} { self.classNames[cls]} {conf}', (x1,y1-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,232,255), 1)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0,232,255), 2)
                    self.countCar(cx,cy,int(Id),cls,image)


    def polylinesCar(self,image,area):
        colorArea = (240,2,3)
        line_width = 2
        radius = 1

        if self.numPoint == 5:
            cv2.polylines(image,[np.array(area,np.int32)],True,colorArea,2)

        if self.numPoint == 1:           
            cv2.circle(image, area[0], radius, colorArea, line_width)

        elif self.numPoint == 2:
            cv2.polylines(image,[np.array(area,np.int32)],True,colorArea,2)

        elif self.numPoint == 3:
            cv2.line(image, area[0], area[1], colorArea, line_width)
            cv2.line(image, area[1], area[2], colorArea, line_width)

        elif self.numPoint == 4:         
            cv2.polylines(image,[np.array(area,np.int32)],True,colorArea,2)
        
        
        if len(self.totalArea) > 0:
            for i in range(len(self.totalArea)):
                cv2.polylines(image,[np.array(self.totalArea[i][2],np.int32)],True,colorArea,2)
               
    def countCar(self,cx,cy,Id,cls,image):
        colorText = (240,2,3)
        for i in range(len(self.totalCars)):
            results = cv2.pointPolygonTest(np.array(self.totalArea[i][2],np.int32),((cx,cy)),False)
            if results > 0:
                
                self.totalCars[i][0].add(Id)
                if self.classNames[cls] == "car":
                    self.totalCars[i][1].add(Id)
                elif self.classNames[cls] == "truck":
                    self.totalCars[i][2].add(Id)
                elif self.classNames[cls] == "bus":
                    self.totalCars[i][3].add(Id)
                elif self.classNames[cls] == "motorbike":
                    self.totalCars[i][4].add(Id)

            if len(self.areaName) >0:
                cv2.putText(image, f'Area: {self.areaName[i]}', (self.totalArea[i][1][0],self.totalArea[i][1][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colorText, 2)

            cv2.putText(image, f'Total={len(self.totalCars[i][0])}', (self.totalArea[i][1][0],self.totalArea[i][1][1]+32), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colorText, 2)
            cv2.putText(image, f'Car={len(self.totalCars[i][1])}', (self.totalArea[i][1][0],self.totalArea[i][1][1]+52), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
            cv2.putText(image, f'Truck={len(self.totalCars[i][2])}', (self.totalArea[i][1][0],self.totalArea[i][1][1]+72), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
            cv2.putText(image, f'Bus={len(self.totalCars[i][3])}', (self.totalArea[i][1][0],self.totalArea[i][1][1]+92), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            cv2.putText(image, f'Motorbike={len(self.totalCars[i][4])}', (self.totalArea[i][1][0],self.totalArea[i][1][1]+112), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,232,255), 2)
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing App...')