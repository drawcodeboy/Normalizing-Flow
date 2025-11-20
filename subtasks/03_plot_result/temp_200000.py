# 200,000 parameter updates result (flow length = 0, 10, 20)
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10, 20]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([72, 80])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [79.371865, 75.153335, 73.890343]
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

    results = [72.219514, 68.567388, 67.709976]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([5, 7.5])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [7.152352, 6.585947, 6.180367]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4_temp_200000.jpg", dpi=500)

if __name__ == '__main__':
    main()