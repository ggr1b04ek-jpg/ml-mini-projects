import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

alpha_prior = 2
beta_prior = 18

n_A = 200
k_A = 15
n_B = 200
k_B = 28

alpha_post_A = alpha_prior + k_A
beta_post_A = beta_prior + (n_A - k_A)

alpha_post_B = alpha_prior + k_B
beta_post_B = beta_prior + (n_B - k_B)

p_values = np.linspace(0, 0.3, 500)

post_A = stats.beta.pdf(p_values, alpha_post_A, beta_post_A)
post_B = stats.beta.pdf(p_values, alpha_post_B, beta_post_B)

plt.figure(figsize=(10, 6))
plt.plot(p_values, post_A, label='Posterior A', color='blue', linewidth=2)
plt.plot(p_values, post_B, label='Posterior B', color='orange', linewidth=2)
plt.axvline((alpha_post_A - 1) / (alpha_post_A + beta_post_A - 2))
plt.axvline((alpha_post_B - 1) / (alpha_post_B + beta_post_B - 2))

plt.title('Байесовский A/B-тест')
plt.xlabel('конверсия (p)')
plt.ylabel('Плотность вероятности')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 0.3)
plt.show()

samples_A = stats.beta.rvs(alpha_post_A, beta_post_A, size=100_000)
samples_B = stats.beta.rvs(alpha_post_B, beta_post_B, size=100_000)

prob_B_better = (samples_B > samples_A).mean()

print(f"Версия A: {k_A} покупок из {n_A} (конверсия {k_A/n_A:.1%})")
print(f"Версия B: {k_B} покупок из {n_B} (конверсия {k_B/n_B:.1%})")
print(f"\nВероятность того, что версия B лучше версии A: {prob_B_better:.1%}")

if prob_B_better > 0.95:
    print("Рекомендация: Внедряйте версию B! (уверенность > 95%)")
elif prob_B_better > 0.80:
    print("Версия B скорее всего лучше, но нужны ещё данные.")
else:
    print("Недостаточно данных для уверенного вывода.")