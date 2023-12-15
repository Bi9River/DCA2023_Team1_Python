from matplotlib import pyplot as plt

def draw_bar_chart(x, y, xlabel, ylabel, img_name):
    fig, ax = plt.subplots()
    fig.clf()
    ax.bar(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(f"{img_name}.png", dpi=500)

if __name__ == "__main__":
    x = [] # TODO x-axis values
    y = [] # TODO y-axis values (accuracy)
    xlabel = "" # TODO x-axis label
    ylabel = "" # TODO y-axis label
    img_name = "" # TODO image name
    draw_bar_chart(x, y, xlabel, ylabel, img_name)
