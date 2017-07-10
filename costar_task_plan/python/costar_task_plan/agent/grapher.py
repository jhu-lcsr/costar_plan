import matplotlib.pyplot as plt

class Grapher:

	def __init__(self, capacity=2000, xaxistitle="Episode", yaxistitle="Reward", title="Reward Per Episode", identifier=0, rolling=False):
		self.samples = []
		self.sampleNums = []
		self.capacity = capacity
		self.sample_rate = 1
		self.samples_processed = 0
		self.xaxistitle = xaxistitle
		self.yaxistitle = yaxistitle
		self.title = title
		plt.ion()
		self.id = identifier
		self.rolling = rolling

	def doubleRate(self):
		temp = []
		tempnums = []
		for i in xrange(len(self.samples)):
			if i%2 == 0:
				temp.append(self.samples[i])
				tempnums.append(self.sampleNums[i])
		self.samples = temp
		self.sampleNums = tempnums
		self.sample_rate *= 2

	def addSample(self, sample):
		self.samples_processed += 1

		if self.rolling:
			self.sampleNums.append(self.samples_processed)
			self.samples.append(sample)
		elif self.samples_processed % self.sample_rate == 0:
			self.sampleNums.append(len(self.samples)*self.sample_rate)
			self.samples.append(sample)

		if len(self.samples) >= self.capacity:
			if self.rolling:
				self.samples.pop(0)
				self.sampleNums.pop(0)
			else:
				self.doubleRate()

	def displayPlot(self, plotstr='b-', ylim=None):
		#'''
		plt.figure(self.id)
		plt.clf()
		plt.plot(self.sampleNums,self.samples, plotstr)
		plt.xlabel(self.xaxistitle)
		plt.ylabel(self.yaxistitle)
		plt.title(self.title)
		if ylim:
			plt.ylim(ylim)
		plt.pause(0.01)
		#'''

	def savePlot(self, fname):
		plt.savefig(fname)