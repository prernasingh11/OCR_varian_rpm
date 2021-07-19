# Optical Character Recognition for Varian Real-time Position Management system

The Varian Real-time Position Management (RPM) system is a video-based system that
compensates for target motion, enabling improved treatment in cancer by tracking the position
of a reflective marker placed on the patient in three dimensions (vertical, longitudinal and
lateral). The RPM system can detect unexpected motion, providing additional confidence that
the target is always accurately positioned for treatment, so the planned dose is delivered to the
tumor. If there is movement, the RPM system detects the interruption and instantly gates the
beam off. The RPM system uses an infrared tracking camera and a reflective marker placed on
the patient and measures the patient’s respiratory pattern and range of motion. It displays them
as three waveforms (vertical, longitudinal, and lateral.) These will be smooth given that the
beam is stopped if there is any patient movement.

In order to determine the position of the Varian Infrared Block on the true beam radiotherapy
system when the couch is moving, we take a video of the signal coming from the Infrared
tracking camera of the RPM system. The camera should be perpendicular to the system and
taken with a tripod to minimize noise due to movement. From there, we use computer vision
techniques, machine learning based optical character recognition (OCR), and develop anomaly
detection techniques to reproduce the three waveforms described above. Due to FDA approval
limitations, currently only 30 seconds of these waveforms can be extracted directly from the
RPM system. With our method, an unlimited time period of movement can be analyzed and
extracted.


# How to install and use this repo

1. Install dependencies:

  pip3 install cv2
  
  pip3 install pytesseract
  
  pip3 install matplotlib
  
  pip3 install pandas
  
  pip3 install numpy
  
  pip3 install scipy
  

2. for mac installation: brew install tesseract or for ubuntu/linux installation: sudo apt-get install tesseract-ocr

3. git pull this repo, or can be manually downloaded through github web interface

Usage:
go to directory that contains interactive_python_ocr.py. The path should end with /OCR_varian_rpm/src/varian_rpm_ocr

Command syntax: python3 interactive_python_ocr.py path_to_input_video output_path min max

# What the package does 

The user is given a snapshot of the video to determine which sections of the image the downstream analyses should be executed on by selecting three boxes (lateral, vertical, and horizontal movement). Then at every 22ms of the video, three static images are extracted based on the coordinates provided by the user. These represent the position of the patient laterally, vertically, and horizontally every 22ms. Each image is then resized to improve image quality and converted to grayscale. Then a gaussian blur filter is applied and the images are thresholded to black and white. Since the numbers are white on a black background, the image is inverted as the OCR is significantly more accurate with a white background, as that was how it was trained. We further erode and dilate the image to improve the quality of the image.
 
Then the Tesseract-OCR Engine is used to predict the value shown in the image. These are easily identifiable by the human eye in most cases but can be challenging in frames where the number is changing. Whenever a number is not predicted by the Tesseract-OCR Engine, rather than moving to the next 22ms frame, the next millisecond is analyzed until a number is returned until the next 22ms snapshot.
 
Next is the anomaly detection aspect of the method. The user can input the minimum and maximum values that should be predicted. If values are given, any value outside of that range will be removed from the waveform. Next, sharp peaks and valleys in the waveforms are detected and all points within +/- 50 ms are removed in case there are repeated anomalous values right next to each other. Finally, in some cases, the numbers predicted by the Tesseract-OCR Engine are correct, but the sign is incorrect. For example, a negative sign is dropped. To account for this, we compare the value for each time point with the value for the time point after it. If the absolute value of difference between the two values is less than the later value squared minus delta, an empirically evaluated threshold, the prior time point is removed.

xi-xi+1 <2*(xi+1) -  delta

Where xi  is the value of the waveform at time i and xi+1 is the value of the waveform at time point immediately after time point i. We determined empirically that the value of 2 for delta works very well.
 
We repeat this “sign correction” five times for each waveform.

# References
1. Real-time Position Management™ System
2. Soille P. (1999) Erosion and Dilation. In: Morphological Image Analysis. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-662-03939-7_3
3. R. Smith, "An Overview of the Tesseract OCR Engine," Ninth International Conference on Document Analysis and Recognition (ICDAR 2007), 2007, pp. 629-633, doi: 10.1109/ICDAR.2007.4376991.





