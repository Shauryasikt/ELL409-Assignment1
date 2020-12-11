import numpy as np
import pandas as pd
import random
import sys, itertools
import scipy.stats
import matplotlib.pyplot as plt

#LDA
class LinearDiscriminantAnalysis:
	
	def __init__(self,data,n_dim):
		self.data = data
		self.n_dim = n_dim
		self.col_label = len(self.data[0])-1
		self.dim = self.col_label
		random.shuffle(self.data)
		self.train_data=self.data[:int(len(self.data)*0.8)]
		self.test_data=self.data[int(len(self.data)*0.8):]
		self.grp_by_class()
		self.calc_means()
		self.calc_SB_SW()
		self.calc_eigenvals()
		self.transform_data()
		self.test_algo()
		self.normal_graph()
		self.flat_graph()

	def grp_by_class(self):
		self.grp_data={}
		for i in self.train_data:
			if i[self.col_label] in self.grp_data:
				self.grp_data[i[self.col_label]].append(i[:self.col_label])
			else:
				self.grp_data[i[self.col_label]]=[i[:self.col_label]]
		self.n_classes = len(self.grp_data)

	def calc_means(self):
		self.class_mean={}
		self.ovr_mean=np.array([0.0 for x in range(self.dim)])
		for i in self.grp_data:
			self.class_mean[i]=np.array([0.0 for x in range(self.dim)])
			for j in self.grp_data[i]:
				for k in range(len(j)):
					self.class_mean[i][k]+=j[k]
					self.ovr_mean[k]+=j[k]
		for i in self.class_mean:
			for j in range(len(self.class_mean[i])):
				self.class_mean[i][j]/=len(self.grp_data[i])
		for i in range(len(self.ovr_mean)):
			self.ovr_mean[i]/=len(self.train_data)

	def calc_SB_SW(self):
		self.SB = np.zeros((self.dim,self.dim))
		for i in self.class_mean:
			mk_m = np.array([self.class_mean[i]-self.ovr_mean]) 
			nk = len(self.grp_data[i])
			aux = ((mk_m.T)*nk).dot(mk_m)
			self.SB += aux
		self.SW = np.zeros((self.dim,self.dim))
		for i in self.class_mean:
			mk = np.array(self.class_mean[i])
			for j in self.grp_data[i]:
				xnk = np.array(j)
				xnk_mk = np.array([xnk-mk])
				self.SW += (xnk_mk.T).dot(xnk_mk)

	def calc_eigenvals(self):
		matrix = np.dot(np.linalg.pinv(self.SW),self.SB)
		eigvals, eigvecs = np.linalg.eig(matrix)
		eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
		eiglist = sorted(eiglist, key = lambda x: x[0], reverse = True)
		w = np.array([eiglist[i][1] for i in range(self.n_dim)])
		self.w = w

	def transform_data(self):
		self.transformed_grp_data = {}
		for i in self.grp_data:
			if i not in self.transformed_grp_data:
				self.transformed_grp_data[i]=[]
			for j in self.grp_data[i]:
				self.transformed_grp_data[i].append(self.w.dot(j)[0])
		self.class_vals = {0:{}, 1:{}}
		for i in self.transformed_grp_data:
			vals = np.array(self.transformed_grp_data[i])
			self.class_vals[i]['mean'] = vals.mean()
			self.class_vals[i]['std'] = vals.std()
		a = self.class_vals[0]['std']**2 - self.class_vals[1]['std']**2
		b = 2*(((self.class_vals[0]['std']**2)*self.class_vals[1]['mean']) - ((self.class_vals[1]['std']**2)*self.class_vals[0]['mean']))
		c = ((self.class_vals[0]['std']**2)*(self.class_vals[1]['mean']**2)) - ((self.class_vals[1]['std']**2)*(self.class_vals[0]['mean']**2)) - 2*((self.class_vals[0]['std']*self.class_vals[1]['std'])**2)*(np.log((self.class_vals[0]['std'])/(self.class_vals[1]['std'])))
		dis = (b**2) - (4*a*c)
		dis = dis**(1/2)
		x1 = ((-b)+dis)/(2*a)
		x2 = ((-b)-dis)/(2*a)
		self.th = 0.0
		if ((self.class_vals[0]['mean']<=x1) and (x1<=self.class_vals[1]['mean'])) or ((self.class_vals[1]['mean']<=x1) and (x1<=self.class_vals[0]['mean'])):
			self.th = x1
		elif ((self.class_vals[0]['mean']<=x2) and (x2<=self.class_vals[1]['mean'])) or ((self.class_vals[1]['mean']<=x2) and (x2<=self.class_vals[0]['mean'])):
			self.th = x2
		self.class_lt_th = -1
		self.class_gt_th = -1
		if ((self.class_vals[0]['mean']<=self.th) and (self.th<=self.class_vals[1]['mean'])):
			self.class_lt_th = 0
			self.class_gt_th = 1
		else:
			self.class_lt_th = 1
			self.class_gt_th = 0

	def test_algo(self):
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		for i in self.test_data:
			pt = np.array([i[x] for x in range(len(i)-1)])
			given_class = i[len(i)-1]
			t_pt = self.w.dot(pt)
			if(t_pt<=self.th):
				pred = self.class_lt_th
			else:
				pred = self.class_gt_th
			if given_class==1 and pred==1:
				tp+=1
			elif given_class==1 and pred==0:
				fn+=1
			elif given_class==0 and pred==1:
				fp+=1
			else:
				tn+=1
			if (tp+fp)!=0:
				self.precision = float(tp)/float(tp+fp)
			else:
				self.precision = 0
			if (tp+fn)!=0:
				self.recall = float(tp)/float(tp+fn)
			else:
				self.recall = 0
			if (self.precision+self.recall)!=0:
				self.f_score = float(2*self.precision*self.recall)/float(self.precision+self.recall)
			else:
				self.f_score = 0
			self.accuracy = float(tp+tn)/float(tp+tn+fp+fn)

	def normal_graph(self):
		mean_0 = self.class_vals[0]['mean']
		mean_1 = self.class_vals[1]['mean']
		std_0 = self.class_vals[0]['std']
		std_1 = self.class_vals[1]['std']

		x = np.linspace(0.0, 300.0, 100)
		y_0 = scipy.stats.norm.pdf(x, mean_0, std_0)
		y_1 = scipy.stats.norm.pdf(x, mean_1, std_1)

		plt.plot(x,y_0,'r')
		plt.plot(x,y_1,'b')
		plt.grid()
		plt.xlim(0.0,300.0)
		plt.xlabel('x')
		plt.show()

	def flat_graph(self):
		y0=[]
		y1=[]
		x0=[]
		x1=[]
		for i in self.transformed_grp_data:
			if i==0:
				for j in self.transformed_grp_data[i]:
					y0.append(1)
					x0.append(j)
			else:
				for j in self.transformed_grp_data[i]:
					y1.append(1.2)
					x1.append(j)

		plt.scatter(x0,y0,color='red')
		plt.scatter(x1,y1,color='blue')
		plt.show()