import matplotlib.pyplot as plt
from math import log

x_as = [2, 4, 10, 100, 1000]
y1 = [9.657, 8.28, 7.339, 6.902, 6.839]
y2 = [9.644, 8.193, 7.399, 6.9, 6.839]
y3 = [9.609, 8.226, 7.38, 6.863, 6.867]

plt.plot([log(x) for x in x_as], y1, label='CMC')
plt.plot([log(x) for x in x_as], y2, label='proportional')
plt.plot([log(x) for x in x_as], y3, label='optimal')
plt.xlabel('log(n) - logarytm naturalny liczby punktów')
plt.ylabel('cena opcji')
plt.title('Zmiana ceny opcji azjatyckiej w zależności od liczby punktów')
plt.legend()
plt.show()
