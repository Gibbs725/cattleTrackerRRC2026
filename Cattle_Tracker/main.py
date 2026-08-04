import cv2
import os
import pandas as pd
import numpy as np
from sympy.printing.pretty.pretty_symbology import line_width
from ultralytics import YOLO
from ultralytics import solutions
import threading

#Setup Variables

Current_Entry_Track= None
Current_CHLOW_Track= None
Current_CHHI_Track= None
frame_counter = 0
latest_EID= None
local_cow=None
tracked_animals = {}


#Reading from the Excel Sheet#
#pd.read_excel("files/2026-06-26_AnimalUpload.xlsx")
df=pd.read_excel("files/2026-06-26_AnimalUpload.xlsx",
                 header=1)
eid_df = df[df["EID*"].notna()] #Filters out "isna" from "notna", obviously
longID=124000192476682                      #DEBUG'D


#RFID MultiThread
def rfid_listener():
    global latest_EID
    global local_cow
    while True:
        try:
            eid=int(input("Enter EID: "))
            latest_EID = eid
            # eid = 124000192476682 #int(input("Enter EID: "))
            print("COW: ", eid)
            Cow_ID = eid_df[eid_df["EID*"].astype(int) == eid]
            if Cow_ID.empty:
                print("No EID found")
                continue
            cow = Cow_ID.iloc[0]
            local_cow = cow
            for field, value in cow.items():
                print(f"{field:<30}: {value}")
            continue
        except ValueError:
            print("Invalid EID")
threading.Thread(target=rfid_listener, daemon=True).start()
#End RFID MultiThreader


#while True:
#    try:
#        #eid = 124000192476682 #int(input("Enter EID: "))
#        #print("COW: ", eid)
#        Cow_ID = eid_df[eid_df["EID*"].astype(int) == eid]
#        if Cow_ID.empty:
#            print("No EID found")
#            continue
#        cow = Cow_ID.iloc[0]
#        for field, value in cow.items():
#            print(f"{field:<30}: {value}")
#        break
#    except ValueError:
#        print("Invalid EID")
#End Reading from the Excel Sheet#

#Logging stuff#



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
    conf= 0.50,
    show_boxes=True,
    show_labels=True
    )


#Camera Loop
while cap1.isOpened():
    ret, frame = cap1.read()
    if not ret:
        print("Video frame is empty or processing is complete.")
        break

#Camera/CV Related
    frame_counter+=1
    results = Pen_Regions(frame)
    #print (results)
    #print (results.total_tracks)
    #print (results.region_counts)
    #print (dir(results))
    track_results= model.track(frame, persist=True, classes=[0,16,19]) #0 is Human
    print(track_results[0].boxes.cls)
    print(track_results[0].boxes.id)
    print(len(track_results[0].boxes))
    print(track_results[0].boxes.data)
    tracks_to_remove=[]

#Checks for the Global RFID input
    if latest_EID is not None:
        print("New EID: ", latest_EID)

#Protects against "No Detect" issues
    if len(track_results)>0:
        BoundBox = track_results[0].boxes
        if len(BoundBox)>0:
            track_ids=BoundBox.id
            print("Boxes:", len(BoundBox))
            print("IDs:", BoundBox.id)
            if len(BoundBox)>0:
                print("Classes: ", BoundBox.cls)

            if track_ids is not None:
                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id.item())
# Creates New Track ID
                    cls = int(track_results[0].boxes.cls[i].item())
                    if track_id not in tracked_animals:
                        tracked_animals[track_id] = {
                            "State": "Unknown",
                            "Location": "Unknown",
                            "LastSeen": frame_counter,
                            "Class":cls
                        }
                    #Debug

                    print(
                        "Track: ", track_id,
                        "Class: ", cls,
                        "In Registry: ", track_id in tracked_animals
                    )
                    #
                    # Custom Bounding Boxes
                    BoundBox = track_results[0].boxes.xyxy[i]
                    #Debug
                    #print("Track: ", track_id, "Exists: ", track_id in tracked_animals)
                    #
                    x1, y1, x2, y2 = map(int, BoundBox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                    if track_id in tracked_animals and "AnimalID" in tracked_animals[track_id]:
                        label =(
                            f"Track:{track_id}"
                            f"ID:{tracked_animals[track_id]['AnimalID']}"
                            f" Gen:{tracked_animals[track_id]['Stock']}")
                    else:
                        label = f"Track: {track_id}"
                    print(label)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                    print(track_results[0].boxes.cls)
#Sets "Freshness"
                tracked_animals[track_id]["LastSeen"] = frame_counter
#Cleanup old tracks
                for old_track in tracked_animals:
                    if frame_counter - tracked_animals[old_track]["LastSeen"] > 30:
                        tracks_to_remove.append(old_track)
                for old_track in tracks_to_remove:
                    print("deleting track: ", old_track)
                    del tracked_animals[old_track]
                    if old_track== Current_Entry_Track:
                        Current_Entry_Track= None
#End Cleanup
#Sets Default
                else:
                    if tracked_animals[track_id]["State"] == "Unknown":
                        tracked_animals[track_id]["State"] = "Detected"
#Event Transitions (for TESTPEN)
                current_testpen = results.region_counts["TESTPEN"]
                if current_testpen > 1:
                    print("MultipleCows")
                else:
                    if previous_testpen == 0 and current_testpen == 1:
                        print("Track ID ", track_id, "Entered TESTPEN")
                        Current_Entry_Track= track_id
                        tracked_animals[track_id]["Location"] = "TESTPEN"
                    if previous_testpen == 1 and current_testpen == 0:
                        print("Track ID ", track_id, "Exited TESTPEN")
                        Current_Entry_Track= None
                        tracked_animals[track_id]["Location"] = "Unknown"
                previous_testpen = current_testpen
#Event Transitions (for TESTPEN) end
#"Processing" EID and Tracking Assignment
                if latest_EID is not None and Current_Entry_Track is not None:
                    tracked_animals[Current_Entry_Track]["EID"] = latest_EID
                    tracked_animals[Current_Entry_Track]["VID"] = local_cow["VID*"]
                    tracked_animals[Current_Entry_Track]["AnimalID"] = local_cow["Animal ID"]
                    tracked_animals[Current_Entry_Track]["Breed"]= local_cow["Breed"]
                    tracked_animals[Current_Entry_Track]["State"] = "Classified"
                    tracked_animals[Current_Entry_Track]["Stock"] = local_cow["Stock Class"]
                    print(
                        f"Attached EID: {latest_EID}"
                        f"To Track ID: {Current_Entry_Track}"
                    )
                    latest_EID = None

#Clears the Global EID ready for the next scan input
                print(track_id)
                print(tracked_animals)
                print("frame: ", frame_counter)
    else:
        pass
#Escape
    #print(type(results.plot_im))
    #print(results.plot_im.shape)
    #print((results.plot_im == frame.all))
    cv2.imshow(Camera, frame)
    # tracked_animals[track_id] = animal_record
    # print(track_results[0].boxes.id)
    #print(track_results[0].boxes.cls)
    #print(track_results[0].boxes.conf)

    #print(dir(track_results[0]))
    #g, h, i, j = results.region_counts.values()
    #print("Pen_1", g, "Pen_2", h, "Pen_3", i, "TestPen", j)
    #print("Animals Currently In TESTPEN ", track_id)


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