#!/usr/bin/python
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, freqz

def readacceldata(datafile):
    '''
        usage: readacceldata('filepath')
        returns: a dictionary whose keys are on of count, x1, y1, z1, x2, y2, z2
                 and values are the columns associated with keys in list form
    '''
    thisdata = []
    f = open(datafile,"r")
    for i,line in enumerate(f):
        if i==0:
            headers = line.rstrip("\r\n").split(", ")
        else:
            thisdata.append(line.rstrip("\r\n").split(" , "))
    numentries = i-1
    data = dict()
    for head in headers:
        data[head] = []
    for i in range(numentries):
        for j in range(1,8):
            data[headers[j]].append(int(thisdata[i][j]))
    data.pop("pre")
    data.pop("pst")
    return data

def downsampledata(data,n):
    '''
        usage: downsampledata(data,n) where data is a dictionary as from
               readacceldata and n is the spacing between samples to KEEP
               n = 3 return 1/3 of the data
        returns: the downsampled input dictionary
    '''
    keys = data.keys()
    numentries = set([len(data[key]) for key in keys])
    if len(numentries) is not 1:
        sys.exit("\n\nError: dictionary entries are not of the same length\n\n")

    downdata = dict()
    for key in keys:
        downdata[key] = []
    for i in range(numentries.pop()-1,-1,-1):
        if i%n==0:
            for key in keys:
                downdata[key].append(data[key].pop(i))
    '''            
    # plot the difference in the input and output
    plt.figure(99)
    for i in ['x1','y1','z1']:
        plt.subplot(2,1,1)
        plt.plot(data['count'],data[i],marker)
        plt.subplot(2,1,2)
        plt.plot(newdata['count'],newdata[i],marker)
    plt.show()
    '''         
    return downdata

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = lfilter(b, a, data)
    y = filtfilt(b, a, data)    # filtfilt passes twice, once forward and once backwards, leaving no phase delay
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    cv = np.convolve(a2, window, 'valid')
    rt = np.sqrt(cv)
    return rt

def threewayplot(fig,xdata,ydata,i,ttl="",xlbl="",ylbl=""):
    plt.figure(fig)
    doubledata = zip(xdata,ydata)
    for j,d in enumerate(doubledata):
        plt.subplot(3,1,j+1)
        plt.plot(d[0],d[1],colors[i*3+j])
        plt.xlim([0,max(d[0])])
        plt.ylabel(keys[i*3+j]+" "+ylbl)
    plt.subplot(3,1,1)
    plt.title(ttl)
    plt.subplot(3,1,j+1)
    plt.xlabel(xlbl)

# plot settings
marker = "-"
p = 1                                   # starting index for plots
keys = ["x1","y1","z1","x2","y2","z2"]  # this needs to match the input files. could use data.keys() below, but it wont be in a nice order
colors = [marker+"b",marker+"r",marker+"g",marker+"b",marker+"r",marker+"g"] 


if __name__=="__main__":
    # get inputs
    targetfrequency = float(sys.argv[1])
    fin = sys.argv[2]
    
    # read csv data and downsample as needed for quicker plotting
    data = readacceldata(fin)
    dsindex = 1                             # set to 1 for no downsampling
    data = downsampledata(data,dsindex)
    samplingrate = 128.                     # hz
    samplespacing = dsindex*1/samplingrate  # seconds

    '''
    ## Test Data
    T = 0.05
    nsamples = T * 5000
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    data = dict()
    for key in keys: data[key] = x
    data['count'] = t
    samplingrate = 5000.
    samplespacing = 1/samplingrate
    '''

    # loop over both accelerometer outputs
    for i in range(2):
        # store time,x,y,z values data
        t = data['count']
        x = data[keys[i*3]]
        y = data[keys[i*3+1]]
        z = data[keys[i*3+2]]
        
        # plot the input data
        threewayplot(p,3*[t],[x,y,z],i,"Acceleration","Time, 128hz")
        p+=1
        
        '''
        # bandpass the input signal to nullify the data
        lowcut = 0
        highcut = 50
        ord = 3
        x = butter_bandpass_filter(x, lowcut, highcut, samplingrate, order=ord)
        y = butter_bandpass_filter(y, lowcut, highcut, samplingrate, order=ord)
        z = butter_bandpass_filter(z, lowcut, highcut, samplingrate, order=ord)
        threewayplot(p,3*[t],[x,y,z],i,
            "Nullified Acceleration, Pass Band Range: {0:.3f}hz to {1:.3f}hz".format(lowcut,highcut),
            "Time, 128hz"
            )
        p+=1
        '''

        # fft on the input and plot
        ffreq = np.fft.fftfreq(len(t),samplespacing)          # get the DFT sample frequencies, window size is len(data['count'])
        xfft = np.fft.fft( x-np.mean(x) )                     # subtract the mean, k=0 in DFT
        yfft = np.fft.fft( y-np.mean(y) )
        zfft = np.fft.fft( z-np.mean(z) )
        threewayplot(
            p,
            3*[ffreq],
            [np.sqrt(xfft.real**2+xfft.imag**2),
             np.sqrt(yfft.real**2+yfft.imag**2),
             np.sqrt(zfft.real**2+zfft.imag**2)],
             i,
             "FFT on Acceleration",
             "Frequency, Hz",
             "Amplitude")
        p+=1

        # bandpass the input target frequency
        bandwidth = 1 #hz
        lowcut = targetfrequency-bandwidth/2.
        highcut = targetfrequency+bandwidth/2.
        ord = 6
        xbp = butter_bandpass_filter(x, lowcut, highcut, samplingrate, order=ord)
        ybp = butter_bandpass_filter(y, lowcut, highcut, samplingrate, order=ord)
        zbp = butter_bandpass_filter(z, lowcut, highcut, samplingrate, order=ord)
        threewayplot(
            p,
            3*[data['count']],
            [xbp,ybp,zbp],
            i,
            "Bandpass Acceleration, Range: {0:.3f}hz to {1:.3f}hz".format(lowcut,highcut),
            "Time, 128hz"
            )
        p+=1
        
        xbpfft = np.fft.fft(xbp)
        ybpfft = np.fft.fft(ybp)
        zbpfft = np.fft.fft(zbp)
        threewayplot(
            p,
            3*[ffreq],
            [np.sqrt(xbpfft.real**2+xbpfft.imag**2),
             np.sqrt(ybpfft.real**2+ybpfft.imag**2),
             np.sqrt(zbpfft.real**2+zbpfft.imag**2)],
             i,
             "FFT on Filtered Acceleration",
             "Frequency, Hz",
             "Amplitude")
        p+=1

    
    # save figures in directories corresponding to each input file
    for i in range(1,p):
        plt.figure(i)
        savedir = fin.split(".")[0]
        plt.gcf().savefig(savedir+"/figure"+str(i)+".png", dpi=300)  
    
    #plt.show()



'''
References

butterworth bandpass: http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass

great stack overflow article, main takeaway is use filtfilt not lfilter:
http://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
'''