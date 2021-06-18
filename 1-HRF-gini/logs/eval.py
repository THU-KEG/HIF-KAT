import numpy as np
import glob

txt_files = glob.glob('*.txt')
for f in sorted(txt_files):
    print(f)
    f = open(f)
    lines = f.readlines()
    prec = [float(x.split()[5].replace(',' ,'')) for x in lines]
    rec = [float(x.split()[8].replace(',' ,'')) for x in lines]
    f1 = [float(x.split()[2].replace(',' ,'')) for x in lines]
    print('& %.1f & %.1f & %.1f | prec = %.1f rec = %.1f f1 = %.1f' % (round(np.mean(prec) * 100, 1), round(np.mean(rec) * 100, 1), round(np.mean(f1)  * 100, 1), round(np.mean(prec) * 100, 1), round(np.mean(rec) * 100, 1), round(np.mean(f1)  * 100, 1)))
    print()