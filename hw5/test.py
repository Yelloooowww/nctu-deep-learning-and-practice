import time
import sys

for i in range(101):
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d / %d " % ('='*i, i,100))
    sys.stdout.flush()
    time.sleep(0.25)
