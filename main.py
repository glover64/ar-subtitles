import multiprocessing
from multiprocessing import freeze_support
from typing import Tuple

import cv2

from detection import get_label_and_box
import speech_recognition as sr
import pyttsx3


def audio_process():
    print("starting audio proc", flush=True)
    r = sr.Recognizer()

    # Function to convert text to
    # speech
    def SpeakText(command):
        # Initialize the engine
        engine = pyttsx3.init()
        engine.say(command)
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

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occurred")


def image_process(queues: Tuple[multiprocessing.Queue, multiprocessing.Queue]):
    print("in image proc", flush=True)
    last_frame, last_label_and_box = queues
    while True:
        if last_frame.empty():
            continue
        label_and_box = get_label_and_box(last_frame.get())
        print("Got label and box", label_and_box, flush=True)
        if label_and_box:
            while not last_label_and_box.empty():
                last_label_and_box.get()
            last_label_and_box.put(label_and_box[0])
            print(label_and_box[0])


def display_process(queues: Tuple[multiprocessing.Queue, multiprocessing.Queue]):
    print("in display proc", flush=True)
    last_frame, last_label_and_box = queues
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    count = 0
    existing_box = None
    existing_label = None

    while (True):
        # reading the frame
        ret, frame = cap.read()
        count += 1
        if count % 8 == 0:
            while not last_frame.empty():
                last_frame.get()
            last_frame.put(frame)

        if not last_label_and_box.empty():
            existing_label, existing_box = last_label_and_box.get()

        if existing_box:
            cv2.rectangle(frame, (existing_box[3], existing_box[0]), (existing_box[1], existing_box[2]),
                          color=(255, 0, 0),
                          thickness=2)
            if existing_label:
                cv2.putText(frame, existing_label,
                            (existing_box[3], existing_box[0]),
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
        # displaying the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break

    cap.release()
    cv2.destroyAllWindows()
    # the following is necessary on the mac,
    # maybe not on other platforms:
    cv2.waitKey(1)


if __name__ == '__main__':
    freeze_support()

    # audio_processer =  multiprocessing.Process(target=audio_process)
    # audio_processer.start()
    # audio_processer.join()
    m_last_frame = multiprocessing.Queue()
    m_last_label_and_box = multiprocessing.Queue()
    display_processer = multiprocessing.Process(target=display_process, args=((m_last_frame, m_last_label_and_box),))
    print("starting display", flush=True)
    display_processer.start()
    print("started display", flush=True)
    print("starting image proc", flush=True)
    image_processer = multiprocessing.Process(target=image_process, args=((m_last_frame, m_last_label_and_box),))
    image_processer.daemon = True
    image_processer.start()
    print("started image proc", flush=True)
    print("joining", flush=True)
    display_processer.join()
    print("joined", flush=True)
