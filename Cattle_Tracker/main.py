import cv2
import os
import pandas as pd
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width
from ultralytics import YOLO
from ultralytics import solutions

#Reading from the Excel Sheet#
pd.read_excel("files/2026-06-26_AnimalUpload.xlsx")
df=pd.read_excel("files/2026-06-26_AnimalUpload.xlsx",
                 header=1)
eid_df = df[df["EID*"].notna()] #Filters out "isna" from "notna", obviously
longID=124000192476682                      #DEBUG'D
while True:
    try:
        eid = int(input("Enter EID: "))
        print("COW: ", eid)
        Cow_ID = eid_df[eid_df["EID*"].astype(int) == eid]
        if Cow_ID.empty:
            print("No EID found")
            continue
        cow = Cow_ID.iloc[0]
        for field, value in cow.items():
            print(f"{field:<30}: {value}")
        continue
    except ValueError:
        print("Invalid EID")

#End Reading from the Excel Sheet#



#Camera 1 initialization#
#cam1 = "Camera 1"
#
#v2.namedWindow(cam1)
#ap1 = cv2.VideoCapture(0)
#model = YOLO('yolo26n.pt')
#
#assert cap1.isOpened(), "Error reading camera"
#if cap1.open(0):
#    ret, frame = cap1.read()
#else:
#    ret = False

#cam2 = "Camera 2"
#cv2.namedWindow(cam2)
#cap2 = cv2.VideoCapture(1)
#assert cap2.isOpened(), "Error reading camera 2"
#if cap2.open(1):
#    ret2, frame = cap2.read()
#else:
#    ret2 = False
#
######### Pen Regions for Camera 1
#
#region_points_1 = {
#    "Pen 1": [(0, 0), (0, 100), (640,100), (640, 0)], #TL Corner, BL Corner, TR Corner, BR Corner, from top to bottom, left to right, x0, y0- x640, y480
#    "Pen 2": [(0, 240), (0, 480), (250, 480), (250, 240)]
#}
#
#region_points_2 = {
#    "Pen 3":[(500, 240), (500, 480), (640, 480), (640, 240)]
#}
#
#########
#
#
##w, h, fps = (int(cap1.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
#
##for (x, y, w, h) in region_points_1.values():
#    #print("X: ", x, "Y: ", y, "W: ", w, "H: ", h)
#
#Pen_Regions=solutions.RegionCounter(
#    show=False,
#    region=region_points_1|region_points_2,
#    model="yolo26n.pt",
#    line_width=(1),
#    classes=[0,16,19], #Humans, Dogs, Cows
#    verbose= False,
#    conf= 0.50)
#
##Camera
#while cap1.isOpened():
#    ret, frame = cap1.read()
#    if not ret:
#        print("Video frame is empty or processing is complete.")
#        break
#
#    results = Pen_Regions(frame)
#    #print (results)
#    g, h, i = results.region_counts.values()
#    print("Pen_1", g, "Pen_2", h, "Pen_3", i)
#
#
#    cv2.imshow(cam1, frame)
#
#    ret2, frame2 = cap2.read()
#    cv2.imshow(cam2, frame2)
#
#    #print(results.total_tracks, results.region_counts.values())
#    #for (x, y, z) in results.region_counts.values():
#        #print(x,y,z)
#
#
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#cap1.release()
#cap2.release()
#cv2.destroyAllWindows()





