import face_recognition
import cv2
from PIL import Image
import model

# Demo from face_recognition: it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

class Video_Emotion_Rec:

    def __init__(self):
        self.predictor = model.Predictor()

        self.int_to_emotion = {
                                0:"Angry",
                                1:"Disgust",
                                2:"Fear",
                                3:"Happy",
                                4:"Sad",
                                5:"Surprise",
                                6:"Neutral",
                            }

    def start_video_recognition(self, image_h=48, image_w=48, processing=Image.ANTIALIAS):
        # params: 
        #   image_h, image_w: size of image after rescaling
        #   processing: type of processing to be used when rescaling

        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        process_this_frame = True
        frames_per_process = 4
        frame_count = 4

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if frame_count == frames_per_process:
                frame_count = 0

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)


                emotions= []
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    face = rgb_small_frame[top:bottom, left:right]
                    face_image = Image.fromarray(face).resize((image_h, image_w), processing).convert("L")
                    #face_image.show()
                    emotion = self.predictor.predict(face_image)

                    emotions.append(self.int_to_emotion[int(emotion)])

            frame_count += 1


            # Display the results
            for (top, right, bottom, left), emotion in zip(face_locations, emotions):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, emotion, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vid = Video_Emotion_Rec()
    vid.start_video_recognition()
