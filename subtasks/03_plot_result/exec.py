import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10, 20, 40]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([72, 81])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [93.860907, 93.682114, 93.239948, 93.118923]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (3) -log-likelihood (nats)
    plt.subplot(1, 3, 3)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([65, 74])
    plt.xlabel('Flow length')
    plt.ylabel('-log-likelihood (nats)')

    results = [87.797915, 87.690830, 87.387068, 87.244485]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([5, 7])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [6.062992, 5.991284, 5.852880, 5.874438]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4.jpg", dpi=500)

if __name__ == '__main__':
    main()