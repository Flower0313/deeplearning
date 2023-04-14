import numpy as np

x = []
last_value = -2
for i in range(1, 150):
    now_value = last_value + 0.04
    x.append(now_value)
    last_value = now_value

print(x)
