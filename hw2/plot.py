#!/usr/bin/python3
import matplotlib.pyplot as plt
fo = open("score_out.txt", "r+")
print ("文件名为: ", fo.name)
list = []
for line in fo.readlines():                          #依次读取每行
    list.append(int(line))
fo.close()

plt.plot(list)
plt.ylabel('episode scores')
plt.xlabel('training episodes')
plt.title('scores')
plt.show()
