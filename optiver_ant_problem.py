import numpy as np
import matplotlib.pyplot as plt

move_set = [0,10],[0,-10],[10,0],[-10,0]

def perform_move(coordinates):
    move = move_set[np.random.choice([0, 1, 2, 3])]
    coordinates += np.array(move)

def inside_of_circle(coordinates):
    x, y = coordinates
    if ((x - 2.5) / 30) **2 + ((y - 2.5) / 40) ** 2 < 1:
        return True
    else:
        return False

def forage_simulation():
    x, y = 0, 0
    coordinates = np.array([x, y])
    n = 0
    while inside_of_circle(coordinates):
        n += 1
        perform_move(coordinates)

    return n


N = 10000
time_array = np.zeros(N)
for k in range(N):
    time_array[k] = forage_simulation()

print(np.mean(time_array))

plt.hist(time_array, bins=30)
plt.show()


