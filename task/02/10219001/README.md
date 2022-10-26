# 10219001
Fernanda Adekeu Alif


## materi sebelumnya
+ Persamaan diferensial, predator-prey, lorentz attractor
+ Monte Carlo
+ Runge-kutta orde 4
+ DFT dan FFT
+ Matriks Temperatur
+ Metode Euler
+ Rekursif Faktorial


## materi paling menarik
+ Persamaan Predator-Prey karena dapat mensimulasikan ketika suatu populasi memiliki banyak predator dan ketika sedikit predator


## materi paling membosankan
+ Osilasi Harmonik karena hanya mensimulasikan osilasi yang biasa dikerjakan dengan tulis tangan


## materi yang sudah dipami
+ Predator-Prey
+ Monte Carlo
+ Runge-Kutta orde 4


## materi yang belum dipahami
+ DFT dan FFT
+ Matriks Temperatur
+ Metode Euler
+ Rekursi Faktorial


## contoh program
+ Buat suatu contoh program dalam Python dan sertakan di sini dengan hasil keluarnnya.


```python
# Mendefinisikan library
import numpy as np
import matplotlib.pyplot as plt

def F(t, x, y):
    return (-x + (x * y), y - (x * y))
    
# Mendefinisikan euler method
def eulerMethod(t_0, t_n, num_steps, x_0, y_0):
    h = abs(t_n - t_0)/num_steps
    t = np.linspace(t_0, t_n, num_steps + 1)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    x[0] = x_0
    y[0] = y_0
    for k in range(0, num_steps):
        (dx, dy) = F(t[k], x[k], y[k])
        x[k + 1] = x[k] + h * dx
        y[k + 1] = y[k] + h * dy
    return (x, y)
    
    
# Mendefinisikan Runge Kutta method
def rungeKutta(t_0, t_n, num_steps, x_0, y_0):
    h = abs(t_n - t_0)/num_steps
    t = np.linspace(t_0, t_n , num_steps + 1)
    x = np.zeros(num_steps + 1)
    y = np.zeros(num_steps + 1)
    x[0] = x_0
    y[0] = y_0
    for k in range(0, num_steps):
        (x1, y1) = F(t[k], x[k], y[k])
        (x2, y2) = F(t[k] + h/2, x[k] + x1 * h/2, y[k] + y1 * h/2)
        (x3, y3) = F(t[k] + h/2, x[k] + x2 * h/2, y[k] + y2 * h/2)
        (x4, y4) = F(t[k] + h, x[k] + x3 * h, y[k] + y3*h)
        x[k + 1] = x[k] + h * (x1 + 2 * x2 + 2 * x3 + x4)/6
        y[k + 1] = y[k] + h * (y1 + 2 * y2 + 2 * y3 + y4)/6
    return (x,y)
    
# Input nilai parameter yang dibutuhkan
t_0 = 0
t_n = 100
x_0 = 0.3
y_0 = 2
num_steps = 3000

# Plotting grafik yang diperoleh berdasarkan
# persamaan euler untuk visualisasi predator - prey
plt.figure(1)
(x_pt,y_pt) = eulerMethod(t_0, t_n, num_steps, x_0, y_0)
plt.plot(x_pt,y_pt,color='blue',label='Euler')
plt.title('Grafik Populasi Predator Terhadap Mangsa - Persamaan Euler')
plt.xlabel('Populasi Predator')
plt.ylabel('Populasi Mangsa')
plt.axis([-1, 5, -1, 5])
plt.savefig('Figure (1) Grafik Predator - Prey berdasarkan Persamaan Euler.png')
```

Hasilnya adalah

<img width="230" alt="Predator Prey" src="https://user-images.githubusercontent.com/97975571/197971057-67c7a786-5f45-4add-b2a7-a799bd3a750c.png">



## cara perkuliahan
+ Cara perkuliahan yang diterapkan sedikit membosankan karena kurang praktik dan teori yang dijelaskan kurang terbayang karena materi tidak dapat diterapkan secara langsung.


## topik sistem fisis
+ Pasar saham merupakan suatu sistem yang dapat dibuat sistem fisisnya dan menarik untuk dikaji karena terdapat banyak faktor dan variabel yang memengaruhi pasar saham dengan cara yang tidak terduga dan dapat memberikan dampak yang berbeda-beda sehingga pasar saham dapat dikaji dari berbagai sudut pandang.


## simulasi dan visualisasi
+ Tidak
