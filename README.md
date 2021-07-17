# Optical Character Recognition for Varian Real-time Position Management system

The Varian Real-time Position Management (RPM) system is a video-based system that
compensates for target motion, enabling improved treatment in cancer by tracking the position
of a reflective marker placed on the patient in three dimensions (vertical, longitudinal and
lateral). The RPM system can detect unexpected motion, providing additional confidence that
the target is always accurately positioned for treatment, so the planned dose is delivered to the
tumor. If there is movement, the RPM system detects the interruption and instantly gates the
beam off. The RPM system uses an infrared tracking camera and a reflective marker placed on
the patient and measures the patient’s respiratory pattern and range of motion. It displays them
as three waveforms (vertical, longitudinal, and lateral. 1 ) These will be smooth given that the
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
