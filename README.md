SLOTH, is a method that uses a Long Short Term Memory Recurren Neural Network (LSTM-RNN) as a probabilistic classifier. The probabilities behaviour over time is compared with the expected one to detect gestures. SLOTH have been presented in the paper "Online Human Gesture Recognition Using Recurrent Neural Networks and Wearable Sensors". Refer to this pubblication for further informations.

The repostiroy contains a python implementation of SLOTH able to recognize the six gestures contained in the "gestures_images" folder. 

## Repository content

* src/SLOTH.py file containts the class implementing all the methods required for SLOTH to work;
* src/gesture_recognition.py uses SLOTH methods on accelerometer data acquired throught ROS. Detected gestures are displayed together with the data streem;
* model/LSTMnet.ht is the LSTM-RNN model loaded into SLOTH;

## Authors

* Alessandro Carfì, dept. DIBRIS Università degli Studi di Genova (Italy) [alessandro.carfi@dibris.unige.it](alessandro.carfi@dibris.unige.it)
* Carola Motolese, dept. DIBRIS Università degli Studi di Genova (Italy) [carola.motolese@gmail.com](carola.motolese@gmail.com)
* Barbara Bruno, dept. DIBRIS Università degli Studi di Genova (Italy) [barbara.bruno@unige.it](barbara.bruno@unige.it)
* Fulvio Mastrogiovanni, dept. DIBRIS Università degli Studi di Genova (Italy) [fulvio.mastrogiovanni@unige.it](fulvio.mastrogiovanni@unige.it)
