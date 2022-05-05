"""
This program is aimed to recognised handwritten words through a webcam device.
This pre-alpha version will return the results both on screen and in an external .txt file.
The guesses will continue as long as words are found in the camera field and will be printed one word/line.
"""

import cv2
import pytesseract

# It is required to inform python where tesseract (on which pytesseract depends) can be found for some obscur reasons
pytesseract.pytesseract.tesseract_cmd = "/Users/tonyanciaux/tesseract/build/tesseract"

# Instantiate the frame object from Webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)
webcam.set(4, 480)


def main():

    while True:
        # As long as "_" is True - meaning the webcam is working -, it'll keep on reading the images from Webcam
        _, capture = webcam.read()

        # Converts the image to grayscale
        capture = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)

        # Applying dilation on the threshold image
        # capture = cv2.dilate(capture, rect_kernel, iterations=1)

        # Detecting Text (".image_to_string" was also adviced cause less chaotic
        # but strangely less effective and doesn't show the boxes nor the words on screen, what a shame)
        boxes = pytesseract.image_to_data(capture)
        print(boxes)

        # splits the elements of boxes into lists
        for i, b in enumerate(boxes.splitlines()):
            with open("recognized_words.txt", "a") as file:
                if i != 0:
                    b = b.split()
                    # b is a list in which the 12th is the word that was recognised by pytesseract
                    if len(b) == 12:
                        # those indices are based on the "print(boxes)" above
                        x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                        # Draws the rectangle on the screen based on coordinates from boxes
                        cv2.rectangle(capture, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        # Print the word on the screen based on the word found in b[11]
                        # Note: in cv2, the origin point is always the upper left corner!
                        cv2.putText(capture, b[11], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 100, 255), 2)
                        text = b[11]
                        file.write(text)
                        file.write("\n")

        cv2.imshow("OCR Recognition System - Pre-Alpha Testing", capture)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
