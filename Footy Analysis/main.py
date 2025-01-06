from utils import read_video, save_video
from trackers import Tracker
import cv2
import supervision as sv
from player_ball_assignment.player_ball_assigner import PlayerBallAssigner


selected_id = None

def click_event(event, x, y, flags, params):
   
    global selected_id
    _, temp_id_mapping = params

    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button clicked
        for track_id, bbox in temp_id_mapping.items():
            x1, y1, x2, y2 = map(int, bbox)
            if x1 <= x <= x2 and y1 <= y <= y2:  # Check if click is inside the bounding box
                selected_id = track_id  # Save the selected tracking ID
                print(f"Player {track_id} selected!")
                break

def detect_and_display_with_click(frame, temp_id_mapping):
    """
    frame             -> The video frame (e.g., video_frames[1]) you want to display
    temp_id_mapping   -> dict of {track_id: [x1, y1, x2, y2]} from your 'tracks["players"][frame_index]'
    """
    global selected_id
    selected_id = None  # Reset each time

    # Draw bounding boxes from the precomputed 'temp_id_mapping'
    for track_id, bbox in temp_id_mapping.items():
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Show the frame and attach mouse callback
    cv2.imshow("Select Player", frame)
    cv2.setMouseCallback("Select Player", click_event, (frame, temp_id_mapping))

    # Wait for user to click or press 'q'
    while selected_id is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return selected_id


def main():
    # Read Video
    video_frames = read_video('input_videos/tott.mp4')

    # Start tracker
    tracker = Tracker('models/v5xu.pt')

    tracks  = tracker.get_object_tracks(video_frames, 
                                         read_from_stub=True,
                                         stub_path='stubs/track_stubs.pk1')
    

    #interpolate ball
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    #player on ball
    player_assinger = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assinger.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
    
    frame_index = 5  # Choose frame
    picked_frame = video_frames[frame_index].copy()  
    temp_id_mapping = tracks["players"][frame_index]  

    final_mapping = {}
    for track_id, info in temp_id_mapping.items():
        final_mapping[track_id] = info["bbox"]

    selected_id = detect_and_display_with_click(picked_frame, final_mapping)

    print("Selected player with ID:", selected_id)
    
    #speed


    # Draw Output and Tracks
    output_video_frames = tracker.draw_annotation(video_frames,tracks, focus_player_id=selected_id)


    
    # Save Video
    save_video(output_video_frames, 'output_videos/output_video.avi')
                

if __name__ == '__main__':
    main()