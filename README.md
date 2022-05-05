# OCR_handwriting_recognition

## In Progress

This code is still at its early stages of development and requires more work to improve accuracy. 
It was developped to try and solve the problem of live recognisition of handwritten text on a whiteboard/blackboard/paperboard through the webcam. 

It was suggested as a potential solution for hybrid-configuration meetings (half on site / half online) to automatically update the dashboard of online team - in this case, with the use of Miro. 

Different options were tested first in order to get the best possible results, including training a whole model on the NIST Special Database 19 (1.2 million pictures of characters). 

The most promising results so far were obtained with the opencv_ocr.py file. 
The name comes from the use of packages OpenCV-python and the Tesseract API. 
