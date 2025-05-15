import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from sympy.codegen.cnodes import sizeof

# Истинные коэффициенты
true_coefs = torch.tensor([1., 2., 3.])

# Генерация случайных данных (2000 примеров, 3 признака)
data = torch.randn(2000, 3)

# Размерность данных (для модели)
dim = true_coefs.size(0)  

# Вычисление меток (Bernoulli, используя логиты (coefs * data))
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
# Определение модели
def model(data):
    coefs_mean = torch.zeros(dim)  # Среднее для коэффициентов
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))  # Коэффициенты с нормальным распределением

    # Модель для меток с логитами
    y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)  # Наблюдения
    return y
# Инициализация NUTS ядра с адаптацией шага
nuts_kernel = NUTS(model, adapt_step_size=True)

# Инициализация MCMC
mcmc = MCMC(nuts_kernel, num_samples=500, warmup_steps=300)

# Запуск MCMC для данных
mcmc.run(data)
# Получение выборок для 'beta'
beta_samples = mcmc.get_samples()['beta']

# Среднее по выборкам для 'beta'
mean_beta = beta_samples.mean(0)

print("Estimated coefficients:", mean_beta)
# Построение гистограмм для каждого коэффициента
fig, axes = plt.subplots(1, dim, figsize=(12, 4))

for i in range(dim):
    axes[i].hist(beta_samples[:, i].detach().numpy(), bins=30, alpha=0.7, edgecolor='k')
    axes[i].set_title(f'Posterior distribution of beta_{i+1}')
    axes[i].set_xlabel(f'beta_{i+1}')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
