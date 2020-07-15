import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, epochs, metric, fignum=0):
    train = history.history[metric]
    val = history.history[f"val_{metric}"]

    plt.plot(range(1, epochs+1), train, label=f"Training {metric}")
    plt.plot(range(1, epochs+1), val, label=f"Validation {metric}")
    plt.legend()
    plt.figure(fignum)
    plt.title(f"Training {metric} vs Validation {metric}")


# Samples from a probability distribution, can be tuned with the temperature parameter
# Higher temperature diversifies output, but can be nonsense
# Lower temperature will predict strict output, but won't be as interesting
def sample(prob, temperature=1.0):
    prob = np.array(prob, dtype=float)
    prob = np.log(prob) / temperature
    prob = np.exp(prob)
    prob = prob / np.sum(prob)
    prob = np.random.multinomial(1, prob.flatten(), 1)
    return np.argmax(prob)
