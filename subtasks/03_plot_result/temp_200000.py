# 200,000 parameter updates result (flow length = 0, 10, 20)
import matplotlib.pyplot as plt

def main():
    plt.figure(figsize=(12, 4))
    margin = 3
    flow_length = [0, 10]

    # (1) Free Energy Bound vs Flow Length
    plt.subplot(1, 3, 1)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([90, 100])
    plt.xlabel('Flow length')
    plt.ylabel('Variational Bound (nats)')

    results = [96.520024, 96.033973]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (3) -log-likelihood (nats)
    plt.subplot(1, 3, 3)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([85, 95])
    plt.xlabel('Flow length')
    plt.ylabel('-log-likelihood (nats)')

    results = [90.451113, 90.012914]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    # (2) KL(q;truth) (nats)
    plt.subplot(1, 3, 2)
    plt.grid(zorder=1)
    plt.xlim([0-margin, 80+margin])
    plt.ylim([4, 8])
    plt.xlabel('Flow length')
    plt.ylabel('KL(q;truth) (nats)')

    results = [6.068911, 6.021059]
    plt.plot(flow_length, results, label='NF', color='blue', zorder=2)
    plt.scatter(flow_length, results, marker='s', color='blue', zorder=2)
    plt.legend()

    plt.tight_layout()
    plt.savefig("assets/figure_4_temp_200000.jpg", dpi=500)

if __name__ == '__main__':
    main()