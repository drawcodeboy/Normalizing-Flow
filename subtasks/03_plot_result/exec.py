import matplotlib.pyplot as plt
import matplotlib

# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# matplotlib.rc('text', usetex=True)

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10, 20, 40, 80]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.title(f"Bound F(x)")
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([71.5, 80])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [79.245757, 74.383364, 72.867311, 72.572694, 72.111064]
    plt.plot(flow_length, results, label='NF (Planar)', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (3) -log-likelihood (nats)
    plt.subplot(1, 3, 3)
    plt.title(f"-ln p(x)")
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([65.5, 72])
    plt.xlabel('Flow length')
    plt.ylabel('-log-likelihood (nats)')

    results = [71.435740, 67.540135, 66.435476, 66.265410, 66.060554]
    plt.plot(flow_length, results, label='NF (Planar)', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.title(f"KL(q;truth)")
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([5.9, 8])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [7.810016, 6.843229, 6.431835, 6.307284, 6.050510]
    plt.plot(flow_length, results, label='NF (Planar)', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4.jpg", dpi=500)

if __name__ == '__main__':
    main()