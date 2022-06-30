import matplotlib.pyplot as plt

def plotter(popc_data, yhats, mean, pth):
    fig, axs = plt.subplots()
    
    for hat in yhats:
        axs.plot(range(2,17), hat, color="#0e1111", alpha=0.4)
    
    axs.plot(range(2,17)[::-1], popc_data, color="#0e1111",
                marker="s", markerfacecolor="none", markeredgecolor="#0e1111",
                label="pure POPC: T=303K")
    axs.plot(range(2,17), mean, color="#d11141", marker="o", label="prediction")
    
    
    axs.legend(loc=3)
    axs.set_xlim([1,17])
    
    axs.tick_params(axis='both',which='both',bottom=True,top=True,left=True,right=True,labelleft=True,direction="in",labelsize=12)
    
    
    axs.set_xlabel("Carbon number n", fontsize=14)
    axs.set_ylabel("Order parameter $S(n)$", fontsize=14)
    axs.legend()
    
    fig.set_size_inches(6,4.5)
    fig.set_dpi(300)
    fig.tight_layout(pad=1)
    
    plt.savefig(pth+'2H_order.png',transparent=True),
    #plt.savefig(pth+'2H_order.svg',transparent=True)