# Emotion-Detector

An emotion detection model built using CNN on the FER_2013 dataset. It can classify images into one of five facial expressions- Sad, Angry, Happy, Surprised and Fearful.

The emotion model has reached an accuracy of approximately 90% but is prone to racial bias due to inaccuracies in data. 

To remove bias, the following techniques have been implemented
1. Using Grayscale images (to help prevent skin color as a deciding factor in facial expressions by machines)
2. Applying translations to images in the form of horizontal and vertical shifts

