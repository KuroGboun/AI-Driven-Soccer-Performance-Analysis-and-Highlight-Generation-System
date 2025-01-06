from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance


class Tracker: 
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #predict missing values
        df_ball_positions = df_ball_positions.interpolate()

        #edge case
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            #"referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            print("Class names in YOLO:", cls_names) 
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            #tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                #if cls_id == cls_names_inv['referees']:
                    #tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ =  get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)), #play around
            angle=0.0,
            startAngle=-45,
            endAngle=235, #360 for full circle
            color= color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        # Add the tracker ID text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3  # Adjust for text size
        text_thickness = 1

        # Position the text slightly above the ellipse
        text_position = (x_center, y2 - int(0.5 * width))
        text = f"ID: {track_id}"  # The tracker ID text to display

        cv2.putText(
            frame,
            text,
            text_position,
            font,
            font_scale,
            color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)               
        
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_rect),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10, y-20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0),2)

        return frame

    def draw_annotation(self, video_frames, tracks, focus_player_id: int = None):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #drawing player
            for track_id, player in player_dict.items():
                # Check if this is the focused player
                if focus_player_id is not None and track_id == focus_player_id:
                    # Use a unique color for the focus player, e.g. bright green
                    frame = self.draw_ellipse(frame, player["bbox"], (0, 255, 0), track_id)
                    if player.get('has_ball', False):
                        frame = self.draw_triangle(frame, player["bbox"], (255, 0, 0))
                    else:
                        frame = self.draw_triangle(frame, player["bbox"], (0, 255, 0))


                else:
                    # Draw in red for all other tracks
                    frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

                
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))


            output_video_frames.append(frame)

        return output_video_frames


   