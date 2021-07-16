# # # imports # # #
import cv2 
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import itertools
import time
import scipy
from scipy.signal import find_peaks
import sys

# # # user inputs # # #
    #1st param = video path
    #2nd param = minumum
    #3rd param = maximum
if len(sys.argv)> 2:
    range_param=True
else:
    range_param=False
cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT);
durationInSeconds = float(totalNoFrames) / float(fps)

cap.set(cv2.CAP_PROP_POS_MSEC,20)      # Go to the 1 sec. position
ret,im = cap.read()                   # Retrieves the frame at the specified second
# Select ROI
lat = cv2.selectROI(im, fromCenter=False)
long = cv2.selectROI(im, fromCenter=False)
vert = cv2.selectROI(im, fromCenter=False)
cv2.waitKey(0) # close window when a key press is detected
cv2.destroyAllWindows()
cv2.waitKey(1)
print("done with user input. Do not select any more boxes")

# # # functions # # #
def get_number(image):
    string=pytesseract.image_to_string(image, config='digits').replace('\n', '').replace('\x0c','')
    try:
        return float(string)
    except:
        return ''

def clean(DF,columnname,thresh=50,times=1):
    for i in range(0,times):
        if range_param:
        	df_cleaned=DF[DF[columnname]<float(sys.argv[3])]
        	df_cleaned=df_cleaned[df_cleaned[columnname]>float(sys.argv[2])].reset_index()
        else:
                df_cleaned = DF.reset_index()
        #plt.plot(df_cleaned['Time (ms)'],df_cleaned[columnname],label='original '+ columnname)

        #find peaks and valleys
        peaks, _ = find_peaks(df_cleaned[columnname], height=None,prominence=(0.1,None))
        peakdf=df_cleaned.iloc[peaks,:]
        #plt.scatter(peakdf['Time (ms)'], peakdf[columnname],label='peaks '+columnname,marker='x',color='green')
        valleys, _ = find_peaks(-df_cleaned[columnname], height=None,prominence=(0.1,None))
        valleydf=df_cleaned.iloc[valleys,:]
        #plt.scatter(valleydf['Time (ms)'], valleydf[columnname],label='valleys '+columnname,marker='x',color='red')

        #remove peaks and valleys with 50 ms buffer
        remove=list(valleys)+ list(peaks)
        lenn=len(remove)
        for x in remove[0:lenn]:
            remove.extend(list(range(x-thresh,x+thresh)))

        remove=list(set(remove))
        df_cleaned=df_cleaned[~df_cleaned.index.isin(remove)]
        #print('after removing peaks and valleys:', df3_cleaned.shape)

    #plt.clf()
    #plt.plot(df_cleaned['Time (ms)'],df_cleaned[columnname],label='cleaned '+ columnname)
    #plt.legend()
    #plt.xlabel('Time (ms)')
    #fig = plt.gcf()
    #fig.set_size_inches(18.5, 10.5)
    #plt.savefig(columnname+'cleaned.png')
    #df_cleaned.to_csv(columnname+'cleaned.csv')
    return df_cleaned

dict={'Lateral':[int(lat[1]),int(lat[1]+lat[3]),int(lat[0]),int(lat[0]+lat[2])],'Longitudinal':[int(long[1]),int(long[1]+long[3]),int(long[0]),int(long[0]+long[2])],'Vertical':[int(vert[1]),int(vert[1]+vert[3]),int(vert[0]),int(vert[0]+vert[2])]}

df1=pd.DataFrame()
df2=pd.DataFrame()
df3=pd.DataFrame()

lists=[df1,df2,df3]

start = time.time()

for ind,p in enumerate(dict.keys()):
    qf=[]
    for x in range(0,int(durationInSeconds*1000),22):
        cap.set(cv2.CAP_PROP_POS_MSEC,x)      # Go to the 1 sec. position
        ret,frame = cap.read()                   # Retrieves the frame at the specified second
        if frame is None:
            break
        pick=frame[dict[p][0]:dict[p][1],dict[p][2]:dict[p][3]]
        #print(x)
        line=[x]

        #image processing
        crop = cv2.resize(pick, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(crop,(15,15),0)
        ret3,blackAndWhiteImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = ~blackAndWhiteImage
        kernel_size = (17,17) # should roughly have the size of the elements you want to remove
        kernel_el = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        kernel_el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
        eroded =   cv2.erode(img, kernel_el1, (-1, -1))
        cleaned = cv2.dilate(eroded, kernel_el, (-1, -1))

        #plt.imshow(cleaned)
        #plt.show()

        #print(get_number(cleaned))
        line.append(get_number(cleaned))
        qf.append(line)

        nexxt=x

        while (get_number(cleaned)==''):
            nexxt=nexxt+1
            line=[nexxt]
            if nexxt > x+22:
                break
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC,nexxt)      # Go to the 1 sec. position
                ret,frame = cap.read()                   # Retrieves the frame at the specified second
                if frame is None:
                    break
                pick=frame[dict[p][0]:dict[p][1],dict[p][2]:dict[p][3]]

                #print(nexxt)
                crop = cv2.resize(pick, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(crop,(15,15),0)
                ret3,blackAndWhiteImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                img = ~blackAndWhiteImage
                kernel_size = (17,17) # should roughly have the size of the elements you want to remove
                kernel_el = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                kernel_el1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
                eroded =   cv2.erode(img, kernel_el1, (-1, -1))
                cleaned = cv2.dilate(eroded, kernel_el, (-1, -1))
                #print(get_number(cleaned))
                #plt.imshow(cleaned)
                #plt.show()
                line.append(get_number(cleaned))
                qf.append(line)
    
    lists[ind] = pd.DataFrame(qf, columns = ['Time (ms)',p])
    lists[ind]=lists[ind].replace(r'^\s*$', np.nan, regex=True)

end = time.time()
print(end - start)

x=clean(lists[0],'Lateral',2)

y=clean(lists[1],'Longitudinal',2)

z=clean(lists[2],'Vertical',1)

def remove_giant_spike(series,columnname,times):
    for i1 in range(0,times):
        for i in range(1, len(series)-2):
            if abs(series[columnname][i:i+1].values-series[columnname][i-1:i].values) < abs(series[columnname][i:i+1].values*2-2):
                pass
            else:
                try:
                    series=series.drop(series.index[i])
                except:
                    pass
                #print(series[i:i+1])
    return pd.DataFrame(series)
            
cleanxlateral=remove_giant_spike(x[['Time (ms)','Lateral']],'Lateral',5)
plt.clf()
plt.plot(cleanxlateral['Time (ms)'], cleanxlateral['Lateral'], label='Cleaned Lateral')
plt.xlabel('Time (ms)')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('Lateral'+'.png')
cleanxlateral.to_csv('Lateral'+'.csv')

cleanylong=remove_giant_spike(y[['Time (ms)','Longitudinal']],'Longitudinal',5)
plt.clf()
plt.plot(cleanylong['Time (ms)'], cleanylong['Longitudinal'], label='Cleaned Longitudinal')
plt.legend()
plt.xlabel('Time (ms)')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('Longitudinal'+'.png')
cleanylong.to_csv('Longitudinal'+'.csv')

cleanzvert=remove_giant_spike(z[['Time (ms)','Vertical']],'Vertical',5)
#cleanzvert=pd.DataFrame(scipy.signal.medfilt2d(cleanzvert[['Time (ms)', 'Vertical']]),columns=['Time (ms)', 'Vertical'])
plt.clf()
plt.plot(cleanzvert['Time (ms)'], cleanzvert['Vertical'], label='Cleaned Vertical')
plt.legend()
plt.xlabel('Time (ms)')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('Vertical'+'.png')
cleanzvert.to_csv('Vertical'+'.csv')
