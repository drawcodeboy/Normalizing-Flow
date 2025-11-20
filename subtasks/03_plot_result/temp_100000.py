# 100,000 parameter updates result (flow length = 0, 10, 20)
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10, 20]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([72, 81])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [80.212316, 76.534312, 75.348011]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (3) -log-likelihood (nats)
    plt.subplot(1, 3, 3)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([65, 74])
    plt.xlabel('Flow length')
    plt.ylabel('-log-likelihood (nats)')

    results = [73.487994, 70.251709, 69.439229]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.grid(zorder=0)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([5, 7])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [6.724323, 6.282603, 5.908781]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=1)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4_temp_100000.jpg", dpi=500)

if __name__ == '__main__':
    main()