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
        eid = 124000192476682 #int(input("Enter EID: "))
        print("COW: ", eid)
        Cow_ID = eid_df[eid_df["EID*"].astype(int) == eid]
        if Cow_ID.empty:
            print("No EID found")
            continue
        cow = Cow_ID.iloc[0]
        for field, value in cow.items():
            print(f"{field:<30}: {value}")
        break
    except ValueError:
        print("Invalid EID")
#End Reading from the Excel Sheet#

#Logging stuff#

tracked_animals = {}
#tracked_animals[track_id] = {
#    "EID": cow["EID*"],
#    "Animal_ID": cow["Animal ID"],
#    "VID": cow["VID*"],
#    "Breed": cow["Breed"],
#    "State": "Identified",
#    "Location": "TESTPEN"
#}
#

#Camera 1 initialization#
Camera = "Camera 1"

cv2.namedWindow(Camera)
cap1 = cv2.VideoCapture(0)
model = YOLO('yolo26n.pt')

assert cap1.isOpened(), "Error reading camera"
if cap1.open(0):
    ret, frame = cap1.read()
else:
    ret = False
#End Camera 1 i#

#cam2 = "Camera 2"
#cv2.namedWindow(cam2)
#cap2 = cv2.VideoCapture(1)
#assert cap2.isOpened(), "Error reading camera 2"
#if cap2.open(1):
#    ret2, frame = cap2.read()
#else:
#    ret2 = False

######## Pen Regions for Camera 1

region_points_1 = {
    "Pen 1": [(0, 0), (0, 100), (640,100), (640, 0)], #TL Corner, BL Corner, TR Corner, BR Corner, from top to bottom, left to right, x0, y0- x640, y480
    "Pen 2": [(0, 240), (0, 480), (250, 480), (250, 240)]
}

region_points_2 = {
    "Pen 3":[(500, 240), (500, 480), (640, 480), (640, 240)],
    "TESTPEN":[(240, 240), (500, 240), (500,400),(240,400)]
}
previous_testpen = 0 #should be default zero

########


#w, h, fps = (int(cap1.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#for (x, y, w, h) in region_points_1.values():
    #print("X: ", x, "Y: ", y, "W: ", w, "H: ", h)

Pen_Regions=solutions.RegionCounter(
    show=False,
    region=region_points_1|region_points_2,
    model="yolo26n.pt",
    line_width=(1),
    classes=[0,16,19], #Humans, Dogs, Cows
    verbose= True,
    conf= 0.50)

#Camera
while cap1.isOpened():
    ret, frame = cap1.read()
    if not ret:
        print("Video frame is empty or processing is complete.")
        break

    results = Pen_Regions(frame)
    #print (results)
    #print (results.total_tracks)
    #print (results.region_counts)
    #print (dir(results))
    track_results= model.track(frame, persist=True, classes=[0])
    track_ids = track_results[0].boxes[0].id

    for track_id in track_ids:
        track_id = int(track_id.item())

        if track_id not in tracked_animals:
            tracked_animals[track_id] = {
                "State": "Unknown",
                "Location": "Unknown",
            }
        else: tracked_animals[track_id]["State"] = "Detected"
        # print(track_id)
    # tracked_animals[track_id] = animal_record
    # print(track_results[0].boxes.id)
    #print(track_results[0].boxes.cls)
    #print(track_results[0].boxes.conf)
    current_testpen=results.region_counts["TESTPEN"]
    if current_testpen > 1:
        print("MultipleCows")
    else:
        if previous_testpen == 0 and current_testpen == 1:
            print("Track ID ", track_id, "Entered TESTPEN")
            tracked_animals[track_id]["Location"] = "TESTPEN"
        if previous_testpen == 1 and current_testpen == 0:
            print("Track ID ", track_id, "Exited TESTPEN")
            tracked_animals[track_id]["Location"] = "Unknown"
    previous_testpen=current_testpen
    #print(dir(track_results[0]))
    g, h, i, j = results.region_counts.values()
    #print("Pen_1", g, "Pen_2", h, "Pen_3", i, "TestPen", j)
    #print("Animals Currently In TESTPEN ", track_id)
    print(tracked_animals)

    cv2.imshow(Camera, frame)

    #ret2, frame2 = cap2.read()
    #cv2.imshow(cap2, frame2)

    #print(results.total_tracks, results.region_counts.values())
    #for (x, y, z) in results.region_counts.values():
        #print(x,y,z)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
#cap2.release()
cv2.destroyAllWindows()





