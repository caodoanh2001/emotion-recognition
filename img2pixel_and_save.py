from PIL import Image
import numpy as np
import os
import glob
d = 'G:/object video detection/object detection(video)/CK+48/'
csv_dir = 'csv/'


import os
files = [os.path.join(d, o) for o in os.listdir(d) 
                    if os.path.isdir(os.path.join(d,o))]
a = len(d)

for file in files:
    files_temp = (glob.glob(file+"/*.png"))
    list_temp = []
    for name in files_temp:
        im = np.array(Image.open(name))
        im = im.reshape(1,-1)
        list_temp.append(im)
    list_temp = (np.asarray(list_temp)).reshape(len(files_temp),2304)
    np.savetxt(csv_dir+file[56:]+".csv", list_temp, delimiter=" ", fmt='%d')
    
import os
import glob
import pandas as pd

all_filenames = (glob.glob("fer2013/*.csv"))
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv("fer2013/fer+ck.csv", index=False, encoding='utf-8-sig')

df = pd.read_csv('fer2013/fer+ck.csv', header=None)
ds = df.sample(frac=1)
ds.to_csv('fer2013/fer+ck_rd.csv')