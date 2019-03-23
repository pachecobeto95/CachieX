import matplotlib.pyplot as plt
import numpy as np
import os

def calculateMedia(resultPath, data):
	file = os.path.join(resultPath, data)
	dataList = []
	with open(file, "r") as f:
		for line in f.readlines():
			data = line.split("\n")[0]
			print(data)
			dataList.append(float(data))

	media = sum(dataList)/len(dataList)
	return media

def writeResults(media, result, bw, latency, mode):
	dirname = os.path.dirname(__file__)
	file = os.path.join(dirname, "results", "media_%s_%s"%(bw, latency))
	with open(file, "a") as f:
		f.write("%s %s %s\n"%(media, bw, latency))

labels = ["1Mbit, 100ms", "4Mbit, 20ms", "8Mbit, 10ms"]
dirname = os.path.dirname(__file__)
mode = "cachier"
resultsPath = os.path.join(dirname, "results", mode)

for files in os.listdir(resultsPath):
	bw = files.split("_")[-1]
	latency = files.split("_")[-2]
	media = calculateMedia(resultsPath, files)
	writeResults(media, resultsPath, bw, latency, mode)



