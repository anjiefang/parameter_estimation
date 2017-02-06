import numpy as np
from datetime import datetime
import gzip
import json

class data_factory:
    def __init__(self, batch_num=10):
        self.batch_num = batch_num
        self.data = None

    def random_interval(self, x, mean,std):
        res = 0.0
        while True:
            n = np.random.normal(mean, std)
            if n+x < 1 and n+x > 0:
                res = n
                break
        return res

    def beta_tweets(self, file, startTime=None, endTime=None, size=None):
        # w = gzip.open('/Users/anjiefang/Desktop/IJCAI2017_EXP/data/' + file.split('/')[-1], 'wb')
        if startTime is not None:
            startdate = datetime.strptime(startTime, '%Y-%m-%d-%H-%M')
            enddate = datetime.strptime(endTime, '%Y-%m-%d-%H-%M')
        if file.split('.')[-1] == 'gz':
            print 'Reading GZ file: ' + str(file)
            f = gzip.open(file)
        elif file.split('.')[-1] == 'json':
            print 'Reading file: ' + str(file)
            f = open(file)
        else:
            print 'Cannot Read file: ' + str(file)
        data = []
        for json_string in f:

            try:
                tweet = json.loads(json_string)
            except:
                continue
            # if 'lang' in tweet:
            #     if tweet['lang'] != lang:
            #         continue
            # else:
            #     if 'lang' in tweet['user']:
            #         if tweet['user']['lang'] != lang:
            #             continue
            #     else:
            #         continue
            timestamp = float(tweet['timestamp_ms'])
            date = datetime.fromtimestamp(timestamp / 1000.0)
            if startTime is not None:
                if date < startdate or date > enddate:
                    continue
            data.append(timestamp)
        #     w.write(json_string)
        # w.close()

        print 'Size: ' + str(len(data))
        self.data = np.array(data)
        if size is not None:
            self.data = np.random.choice(data,size)
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        for i in [ii for ii in range(len(self.data)) if self.data[ii] == 0.0]:
            self.data[i] = 0.0 + 1e-4
        for i in [ii for ii in range(len(self.data)) if self.data[ii] == 1.0]:
            self.data[i] = 1.0 - 1e-4
        return self.data

    def beta_samples(self, a=3, b=4, size=20000, sort=True, isAddNoise=False, mean=0, std=0.1):
        self.data = np.random.beta(a,b,size)
        if isAddNoise:
            print 'Noisy added!'
            self.data = np.append(self.data, np.random.beta(a+std, b+std,size))
            self.data = np.append(self.data, np.random.beta(a-std, b-std, size))
            self.data = np.append(self.data, np.random.beta(a+std, b-std, size))
            self.data = np.append(self.data, np.random.beta(a-std, b+std, size))
            self.data = np.random.choice(self.data,size)
            # noise = np.random.normal(mean, std, size)
            # self.data += noise
            # self.data = np.exp(self.data) /( 1+ np.exp(self.data))
            # self.data = (self.data - self.data.min())/(self.data.max()-self.data.min())
            # self.data += 1
            # self.data %= 1
            # self.data = np.array([ n+self.random_interval(n,mean,std) for n in self.data])
        if sort: self.data = sorted(self.data)

    def get_batch(self, batch_index, isCum=True):
        if batch_index > self.batch_num or batch_index < 0:
            raise
        numPerBatch, _ = np.histogram(self.data, bins= self.batch_num)
        if isCum:
            return self.data[:np.cumsum(numPerBatch[:batch_index+1])[-1]]
        else:
            if batch_index == 0:
                return self.data[:numPerBatch[0]]
            else:
                return self.data[np.cumsum(numPerBatch[:batch_index])[-1]:np.cumsum(numPerBatch[:batch_index+1])[-1]]

    def get_data(self):
        return self.data