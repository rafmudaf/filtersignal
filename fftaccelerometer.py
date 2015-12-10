#!/usr/bin/python
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

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
    y = lfilter(b, a, data)
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

def threewayplot(fig,xdata,ydata,i):
    plt.figure(fig)
    doubledata = zip(xdata,ydata)
    for j,d in enumerate(doubledata):
        plt.subplot(3,1,j+1)
        plt.plot(d[0],d[1],colors[i*3+j])
        plt.ylabel(keys[i*3+j])

# plot settings
marker = "-"
p = 1                                   # starting index for plots
keys = ["x1","y1","z1","x2","y2","z2"]  # this needs to match the input files. could use data.keys() below, but it wont be in a nice order
colors = [marker+"b",marker+"r",marker+"g",marker+"b",marker+"r",marker+"g"] 


# main
targetfrequency = int(sys.argv[1])
fin = sys.argv[2]

data = readacceldata(fin)
dsindex = 1                             # set to 1 for no downsampling
data = downsampledata(data,dsindex)
samplingrate = 128.
samplespacing = dsindex*1/samplingrate

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
    # plot the input data
    threewayplot(p,3*[data['count']],
        [data[keys[i*3]],data[keys[i*3+1]],data[keys[i*3+2]]],i)
    p+=1

    # fft on the input and plot
    ffreq = np.fft.fftfreq(len(data['count']),d=samplespacing)
    xfft = np.fft.fft(data[keys[i*3]]);   xfft.real[0] = 0 # remove DC signal
    yfft = np.fft.fft(data[keys[i*3+1]]); yfft.real[0] = 0
    zfft = np.fft.fft(data[keys[i*3+2]]); zfft.real[0] = 0
    
    threewayplot(p,3*[ffreq],
        [np.sqrt(xfft.real**2+xfft.imag**2),
         np.sqrt(yfft.real**2+yfft.imag**2),
         np.sqrt(zfft.real**2+zfft.imag**2)]
         ,i)
    p+=1

    
    # bandpass the input target frequency
    lowcut = targetfrequency*3/4.       # modify these as needed for the band pass range
    highcut = targetfrequency*5/4.
    print "\nBandpass filter\nLow: "+str(lowcut)+"hz\nHigh: "+str(highcut)+"hz\n"

    ord = 3
    xbp = butter_bandpass_filter(data[keys[i*3]], lowcut, highcut, samplingrate, order=ord)
    ybp = butter_bandpass_filter(data[keys[i*3+1]], lowcut, highcut, samplingrate, order=ord)
    zbp = butter_bandpass_filter(data[keys[i*3+2]], lowcut, highcut, samplingrate, order=ord)
    threewayplot(p,3*[data['count']],[xbp,ybp,zbp],i)
    p+=1
    
    xbpfft,ybpfft,zbpfft = np.fft.fft(xbp),np.fft.fft(ybp),np.fft.fft(zbp)
    threewayplot(p,3*[ffreq],
        [np.sqrt(xbpfft.real**2+xbpfft.imag**2),
         np.sqrt(ybpfft.real**2+ybpfft.imag**2),
         np.sqrt(zbpfft.real**2+zbpfft.imag**2)]
         ,i)
    p+=1
plt.show()





'''
## Band pass filter example
# Sample rate and desired cutoff frequencies (in Hz).
fs = 5000.0
lowcut = 500.0
highcut = 1250.0

# Plot the frequency response for a few different orders.
plt.figure(1)
plt.clf()
for order in [3, 6, 9]:
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         '--', label='sqrt(0.5)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend(loc='best')

# Filter a noisy signal.
T = 0.05
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)
a = 0.02
f0 = 600.0
x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
x += a * np.cos(2 * np.pi * f0 * t + .11)
x += 0.03 * np.cos(2 * np.pi * 2000 * t)
plt.figure(2)
plt.clf()
plt.plot(t, x, label='Noisy signal')

y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
plt.xlabel('time (seconds)')
plt.hlines([-a, a], 0, T, linestyles='--')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()
'''



'''
## Low pass filter example
# Filter requirements.
order = 6
fs = 128.       # sample rate, Hz
cutoff = 50     # desired cutoff frequency of the filter, Hz
# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.figure(p); p+=1
plt.subplot(4, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel("Frequency [Hz]")
plt.show()
'''
'''
# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(3, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, 0.5*fs)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)

# Filter the data, and plot both the original and filtered signals.
y = butter_lowpass_filter(data, cutoff, fs, order)

plt.subplot(3, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplot(3,1,3)
x = np.fft.fftfreq(len(t),d=1/30.)
#print x
plt.plot(x,np.sqrt(np.fft.fft(data).imag**2+np.fft.fft(data).real**2))
#plt.plot(x,np.sqrt(np.fft.fft(y).imag**2+np.fft.fft(y).real**2))
plt.xticks(np.arange( min(x), max(x)+1, 1.0))

plt.show()
'''