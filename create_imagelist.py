import os

DIR = './gen_512'
OFILE = 'gen512.txt'

f = open(OFILE, "a")

for root, dirs, files in os.walk(DIR):
    for file in files:
        if ".png" in file:
            fpath = os.path.join(root, file)
            f.write(fpath)
            f.write('\n')
f.close()
