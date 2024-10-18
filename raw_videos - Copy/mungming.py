import cv2
import pandas as pd

def play_sign_gesture(gesture_path):
    cap = cv2.VideoCapture(gesture_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Sign Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def convert_to_sign(entry_widget, output_label_widget):
    text = entry_widget.get()  # Get the text from the entry widget
    label_names_df = pd.read_csv(r"C:\Users\Asus\OneDrive\kosen2\signlent\label_name_jingjing.csv")
    dictionary = label_names_df['meaning'].tolist()
    if text in dictionary:
        label = label_names_df.loc[label_names_df['meaning'] == text, 'index'].values[0]
        label_str = str(label)  # Convert label to string if necessary

        if label_str in sign_dictionary:
            play_sign_gesture(sign_dictionary[label_str])
            output_label_widget.config(text=f"Label: {label_str}")
        else:
            output_label_widget.config(
                text=f"We don't have this sign language translate but you can check it in file 'nslt_2000', Label is: {label_str}"
            )
    else:
        output_label_widget.config(text="Sorry, don't have this text in dictionary")

sign_dictionary = {
    "1222": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\50826.mp4",
    "1610": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\01622.mp4",
    "1217": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\49934.mp4",
    "1215": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\49150.mp4",
    "1192": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\44445.mp4",
    "1210": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\48325.mp4",
    "182": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\03435.mp4",
    "1657": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\09262.mp4",
    "1811": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\33851.mp4",
    "1963": r"C:\Users\Asus\OneDrive\kosen2\signlent\raw_videos\57391.mp4"
}
