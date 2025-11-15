import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10, 20]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([90, 100])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [93.894511, 94.243438, 94.676130]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (3) -log-likelihood (nats)
    plt.subplot(1, 3, 3)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([85, 90])
    plt.xlabel('Flow length')
    plt.ylabel('-log-likelihood (nats)')

    results = [87.810139, 88.124440, 88.399779]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([4, 8])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [6.084372, 6.118998, 6.276352]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4.jpg", dpi=500)

if __name__ == '__main__':
    main()