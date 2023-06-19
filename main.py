
import cv2
import numpy as np
import mediapipe as mp
from collections import deque



# Import docx NOT python-docx


# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
cpoints = [deque(maxlen=1024)]
mpoints = [deque(maxlen=1024)]
blpoints = [deque(maxlen=1024)]



# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
cyan_index =0
magenta_index = 0
black_index = 0

#The kernel to be used for dilation purpose
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (30,1), (90,50), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (100,1), (160,50), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (170,1), (230,50), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (240,1), (300,50), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (310,1), (370,50), (0,255,255), 2)
paintWindow = cv2.rectangle(paintWindow, (380,1), (440,50), (255,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (450,1), (510,50), (255,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (520,1), (580,50), (0,0,0), 2)




cv2.putText(paintWindow, "CLEAR", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (310, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "CYAN", (390, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "MAGENTA", (455, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (30,1), (90,50), (0,0,0), 2)
    frame = cv2.rectangle(frame, (100,1), (160,50), (255,0,0), 2)
    frame = cv2.rectangle(frame, (170,1), (230,50), (0,255,0), 2)
    frame = cv2.rectangle(frame, (240,1), (300,50), (0,0,255), 2)
    frame = cv2.rectangle(frame, (310,1), (370,50), (0,255,255), 2)
    frame = cv2.rectangle(frame, (380,1), (440,50), (255,255,0), 2)
    frame = cv2.rectangle(frame, (450,1), (510,50), (255,0,255), 2)
    frame = cv2.rectangle(frame, (520,1), (580,50), (0,0,0), 2)


    cv2.putText(frame, "CLEAR", (30, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (110, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (310, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "CYAN", (390, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "MAGENTA", (455, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (530, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            cpoints.append(deque(maxlen=512))
            cyan_index += 1
            mpoints.append(deque(maxlen=512))
            magenta_index += 1
            blpoints.append(deque(maxlen=512))
            black_index += 1


        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                cpoints = [deque(maxlen=512)]
                mpoints = [deque(maxlen=512)]
                blpoints = [deque(maxlen=512)]


                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0
                cyan_index =0
                magenta_index =0
                black_index =0


                paintWindow[67:,:,:] = 160
            elif 100 <= center[0] <= 160:
                    colorIndex = 0 # Blue
            elif 170 <= center[0] <= 230:
                    colorIndex = 1 # Green
            elif 240 <= center[0] <= 300:
                    colorIndex = 2 # Red
            elif 310 <= center[0] <= 370:
                    colorIndex = 3 # Yellow
            elif 380 <= center[0] <= 440:
                colorIndex = 4  # Yellow
            elif 450 <= center[0] <= 510:
                colorIndex = 5  # Yellow
            elif 520 <= center[0] <= 580:
                colorIndex = 6  # Yellow

        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
            elif colorIndex == 4:
                cpoints[cyan_index].appendleft(center)
            elif colorIndex == 5:
                mpoints[magenta_index].appendleft(center)
            elif colorIndex == 6:
                blpoints[black_index].appendleft(center)

    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        cpoints.append(deque(maxlen=512))
        cyan_index += 1
        mpoints.append(deque(maxlen=512))
        magenta_index += 1
        blpoints.append(deque(maxlen=512))
        black_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints,cpoints, mpoints, blpoints]
    # for j in range(len(points[0])):
    #         for k in range(1, len(points[0][j])):
    #             if points[0][j][k - 1] is None or points[0][j][k] is None:
    #                 continue
    #             cv2.line(paintWindow, points[0][j][k - 1], points[0][j][k], colors[0], 2)
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    # cv2.imshow("")
    cv2.imshow("Paint", paintWindow)
    if cv2.waitKey(1) == ord('q'):
        break


# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()



print("Now you can speak :-\n")


#Stacking both the things are stacked together

import speech_recognition as sr
import pyttsx3

# Initialize the recognizer
r = sr.Recognizer()


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.save_to_file(command, 'try.mp3')
    engine.runAndWait()


# Loop infinitely for user to
# speak

while (1):

    # Exception handling to handle
    # exceptions at the runtime
    try:

        # use the microphone as source for input.
        with sr.Microphone() as source2:

            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say ", MyText)
            SpeakText(MyText)
            file = open('new.txt', 'w')
            file.write(MyText)
            file.close()

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")

    break;








