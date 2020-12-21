import matplotlib.pyplot as plt
import numpy as np 

def grad(x):
    '''
    Dùng để tính đạo hàm
    '''
    return 2*x+ 5*np.cos(x)

def cost(x):
    '''
    Dùng để tính giá trị của hàm số. 
    '''
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    '''
    Thuật toán Gradient Desent. \n
    Đầu vào của hàm số này là learning rate và điểm bắt đầu. \n
    Thuật toán dừng lại khi đạo hàm có độ lớn đủ nhỏ.
    '''
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

xbar = np.linspace(-6, 8, 1000)
ybar = cost(xbar)


def visualize(x, plot_title):
    i=0
    for i, item in enumerate(x):
        plt.clf()
        plt.title(plot_title)
        plt.plot(xbar, ybar, '-')
        plt.xlabel("Iterator = " + str(i))
        _x, _y = item, cost(item)
        plt.plot(_x, _y, 'ro')
        plt.text(_x, _y, "(%.4f, %.4f)"%(_x, _y))
        plt.draw()
        plt.pause(0.001)

lstLR = []
lstBias = []
lstIterators = []

for bias in range(-5, 6):
    for eta in range(1, 51, 5):
        lr = eta/100
        (x, it) = myGD1(lr, bias)
        print('Bias = %d, η = %f, Solution x = %f, cost = %f, obtained after %d iterations'%(bias, lr, x[-1], cost(x[-1]), it))
        lstBias.append(bias)
        lstLR.append(lr)
        lstIterators.append(it)
        # visualize(x, "Thực nghiệm với bias = %d, η = %.2f"%(bias,lr))
        # plt.title("Thực nghiệm với bias = %d, η = %.2f"%(bias,lr))
        # plt.plot(xbar, ybar, '-')
        # _x, _y = x[-1], cost(x[-1])
        # plt.plot(_x, _y, 'ro')
        # plt.text(_x, _y, "(%.4f, %.4f)"%(_x, _y))
        # plt.xlabel("Bias = " + str(bias) + ", η = " + str(lr) + " => Solution x = " + str(x[-1]) + " sau " + str(it)+ " iterations.")
        # plt.pause(1)

fig = plt.figure()
ax = fig.gca(projection='3d')
my_cmap = plt.get_cmap('bone')
trisurf = ax.plot_trisurf(lstLR, lstBias, lstIterators, cmap = my_cmap, vmax=20, linewidth=0.5, edgecolors='grey')

X, Y, Z = [], [], []
for index, it in enumerate(lstIterators):
    if it<10:
        X.append(lstLR[index])
        Y.append(lstBias[index])
        Z.append(it)

fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 5) 
plt.title('Sự phụ thuộc giữa số vòng lặp (Iterator) để giải thuật hội tụ \n Vào tốc độ học (η) và giá trị khởi tạo (Bias) \n\n') 

x_ticks = np.arange(0, 0.5, 0.05)
ax.set_xticks(x_ticks)
ax.set_xlabel('Learning rate (η)', fontweight ='bold')  

y_ticks = np.arange(-5, 6)
ax.set_yticks(y_ticks)
ax.set_ylabel('Bias', fontweight ='bold')  

z_ticks = np.arange(0, 100, 10)
ax.set_zticks(z_ticks)
ax.set_zlabel('Iterator', fontweight ='bold') 

plt.show()