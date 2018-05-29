import numpy as np

N = 10000
x = np.arange(N)
np.random.shuffle(x)
m = np.mean(x)

k = 50


est_list = []
for _ in range(10000):
    chosen_list = []
    prob = [1.0 for _ in range(len(x))]
    for _ in range(k):
        prob /= np.sum(prob)
        chosen_inx = np.random.choice(range(len(x)), p=prob)
        chosen_list.append(x[chosen_inx])
        prob[chosen_inx] = 0.0

    est_list.append(np.mean(chosen_list))
est = np.mean(est_list)


norm_est_list = []
for _ in range(10000):
    norm_est_list.append(np.mean(np.random.choice(x, size=k, replace=True)))
norm_est = np.mean(norm_est_list)

print("Real mean: {}".format(m))
print("Normal etimated: {}, diff: {}".format(norm_est, m-norm_est))
print("My Estimated: {}, diff: {}".format(est, m-est))