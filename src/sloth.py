#!/usr/bin/python

import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt


class sloth:
    def __init__(self, model_path, window_size, class_size, feature_size, rho, tau, c):
        self.model_path = model_path
        self.window_size = window_size
        self.class_size = class_size
        self.feature_size = feature_size
        self.probabilities_size = window_size

        self.graph = tf.get_default_graph()
        self.model = load_model(self.model_path)
        self.window = np.empty((1,self.window_size,self.feature_size))
        self.window[:] = np.nan

        self.probabilities = np.empty((1,self.probabilities_size,self.class_size))
        self.probabilities[:] = np.nan

        self.rho = rho
        self.tau = tau
        self.c = c

        self.time = 0
        self.peaks = np.zeros((1,self.class_size))

        self.gestures = []

    def classify(self):
        if not np.any(np.isnan(self.window)):
            with self.graph.as_default():
                self.probabilities = np.roll(self.probabilities,self.probabilities_size-1,1)
                self.probabilities[0,-1,:] = self.model.predict(self.window, batch_size=self.window_size, verbose=2)
        else:
            print "The sliding window is not completely full"

    def detect(self):
        delta_prob = (self.probabilities[0,-1,:] - self.probabilities[0,-1-1,:]) 
        possible_peaks = np.where(delta_prob > self.rho)
        possible_peaks = possible_peaks[0]

        for ids in possible_peaks:
            if self.peaks[0, ids] == 0:
                self.peaks[0, ids] = self.time
            else:
                time_diff = self.time - self.peaks[0, ids]
                if time_diff >= self.c[ids]:
                    self.peaks[0, ids] = self.time
        active_peaks = np.where(self.peaks[0,:]> 0)
        active_peaks = active_peaks[0]

        for ids in active_peaks:
            time_diff = self.time - self.peaks[0, ids] + 1
            if time_diff >= self.c[ids]:
                start = int(self.probabilities_size-time_diff)
                prob_mean = np.mean(self.probabilities[0,start:,ids])
                if prob_mean > self.tau[ids]:
                    self.peaks[0, ids] = 0
                    self.gestures.append(ids+1)

    def window_update(self, x, y, z):
        self.window = np.roll(self.window,self.window_size-1,1)
        self.window[:,-1,0] = x
        self.window[:,-1,1] = y
        self.window[:,-1,2] = z
        self.time += 1

    def display(self):
        plt.clf()
        plt.figure(1)
        plt.subplot(911)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,0])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(912)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,1])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(913)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,4])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(914)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,5])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(915)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,2])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])
        plt.subplot(916)
        plt.plot(range(0,self.probabilities_size),self.probabilities[0,:,3])
        plt.axis([0, self.probabilities_size, -0.5, 1.5])

        plt.subplot(917)
        plt.plot(range(0,self.window_size),self.window[0,:,0])
        plt.axis([0, self.window_size, -10, 10])
        plt.subplot(918)
        plt.plot(range(0,self.window_size),self.window[0,:,1])
        plt.axis([0, self.window_size, -10, 10])
        plt.subplot(919)
        plt.plot(range(0,self.window_size),self.window[0,:,2])
        plt.axis([0, self.window_size, -10, 10])
        
        plt.ion()
        plt.pause(0.05)

    def get_gesures(self):
        temp = self.gestures
        self.gestures = []
        return temp