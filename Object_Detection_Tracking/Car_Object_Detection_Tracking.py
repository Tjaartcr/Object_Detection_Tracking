


import cv2
import os
import math

from ultralytics import YOLO

##from object_detection import ObjectDetection

##od = ObjectDetection()

Model_File = "C://Python_Env//New_Virtual_Env//Personal//Weights//yolo-Weights//yolov8n.pt"

print("Loading the Model YoloV8n.....")

# YOLO Model
Obsticle_Detection_Vision_Model = YOLO(Model_File)

with open("C://Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
    classNames = [line.rstrip('\n') for line in f]


##def Vision_Object_Detection_Tracking():

print("Object detection system is running.....")

##Test_Image_Path = ("C://Users//Tjaart AAEEPA//Videos//Object_Trackin_Test_Video//los_angeles.mp4")
Test_Image_Path = ("C://Users//Tjaart AAEEPA//Videos//Object_Trackin_Test_Video//Test_Cars.mp4")
##Test_Image_Path = ("C://Users//Tjaart AAEEPA//Videos//Object_Trackin_Test_Video//fruit_and_vegetable_detection.mp4")

cap = cv2.VideoCapture(Test_Image_Path)

##cap.set(3, 480)
##cap.set(4, 640)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with open("C://Python_Env//New_Virtual_Env//Personal//Object_Detection//Object_Detection_List.txt", "r") as f:
    classNames = [line.rstrip('\n') for line in f]
##    print (classNames)

print("Object detection Model is Loaded.....")

Count = 0

Center_Point_List_Previous = []
Center_Points_Prev_Frame = []
Tracking_Objects = {}
Track_id = 0
Distance = 0
Center_Points_Current_Frame = []
center_points_prev_frame = []
tracking_objects = {}
track_id = 0



while True:
    
    frame, img = cap.read()
    
    results = Obsticle_Detection_Vision_Model(img, stream=False)

##    print("Object Detection Software is running.....")

    Count += 1
    
    Center_Point_List_Current = []

    if not frame:
        break
    
    for r in results:
        boxes = r.boxes

        for box in boxes:
            Track_id += 1
            x, y, w, h = box.xyxy[0]
            x, y, w, h = int(x), int(y), int(w), int(h)

##            print("Frame No. : ", Count, "   ", x, y, w, h)
            
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 1)

            # Draw confidence and class name
            confidence = round(float(box.conf[0]) * 100, 2)
            class_index = int(box.cls[0])
            class_name = classNames[class_index]
            text = f"{class_name}: {confidence}%"
            org = (x, y - 10)  # Place text slightly above the bounding box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 255, 0)
            thickness = 1

            cv2.putText(img, text, org, font, font_scale, color, thickness)

            Cx = int((x + w) / 2)
            Cy = int((y + h) / 2)
            
            My_List = (Cx, Cy)
            Center_Point_List_Current.append((Cx, Cy))

        # Only at the beginning we compare previous and current frame
        if Count <= 2:
            for pt in Center_Point_List_Current:
                for pt2 in center_points_prev_frame:
                    distance = math.hypot(pt[0] - pt2[0], pt[1] - pt2[1])

                    if distance < 40:
                        tracking_objects[track_id] = pt
                        track_id += 1

        else:

            tracking_objects_copy = tracking_objects.copy()
            Center_Point_List_Current_copy = Center_Point_List_Current.copy()

            for object_id, pt2 in tracking_objects_copy.items():
                object_exists = False
                for pt in Center_Point_List_Current_copy:
                    cv2.circle(img, pt, 2, (0, 255, 0), -1) 
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

##                    print("Distance : ", distance)
                    
                    # Update IDs position
                    if distance < 40:
                        tracking_objects[object_id] = pt
                        object_exists = True
                        if pt in Center_Point_List_Current:
                            Center_Point_List_Current.remove(pt)
                        continue

                # Remove IDs lost
                if not object_exists:
                    tracking_objects.pop(object_id)

            # Add new IDs found
            for pt in Center_Point_List_Current:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(img, pt, 2, (0, 0, 255), -1)
            cv2.putText(img, str(object_id), (pt[0], pt[1] - 7), 0, 0.75, (0, 0, 255), 1)
            My_New_String = str(object_id) + str(text)

##        print("Tracking objects")
##        print(tracking_objects)
##
##        print("CUR FRAME LEFT PTS")
##        print(Center_Point_List_Current)

        # Make a copy of the points
        center_points_prev_frame = Center_Point_List_Current.copy()

        cv2.imshow("Face Detection System", img)

        key = cv2.waitKey(1)

        if 0xFF == 27: # Press 'ESC' to exit
            break

            cap.release()
            cv2.destroyAllWindows()

