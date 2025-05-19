import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, HMC

NUM_SAMPLES: int = 500  # Количество выборок
WARMUP_STEPS: int = 300  # Шаги разогрева
DIM: int = 3  # Размерность данных (количество признаков)
BATCH_SIZE: int = 2000  # Количество данных (примеров)

Tensor = torch.Tensor  # Тип для тензоров

# Истинные коэффициенты
true_coefficients: Tensor = torch.tensor([1., 2., 3.])

# Генерация случайных данных (2000 примеров, 3 признака)
data: Tensor = torch.randn(BATCH_SIZE, DIM)

# Размерность данных (для модели)
dim: int = true_coefficients.size(0)

# Метки
labels: Tensor = dist.Bernoulli(logits=(true_coefficients * data).sum(-1)).sample()


# Определение модели
def model(_data: Tensor) -> Tensor:
    coefficients_mean: Tensor = torch.zeros(DIM)  # Среднее для коэффициентов
    coefficients: Tensor = pyro.sample('beta', dist.Normal(coefficients_mean,
                                                           torch.ones(DIM)))  # Коэффициенты с нормальным распределением

    y: Tensor = pyro.sample('y', dist.Bernoulli(logits=(coefficients * _data).sum(-1)), obs=labels)  # Наблюдения
    return y


# Функция для выбора метода сэмплинга
def run_mcmc(method: str) -> MCMC:
    if method == "NUTS":
        # Инициализация NUTS ядра с адаптацией шага
        kernel = NUTS(model, adapt_step_size=True)
    elif method == "HMC":
        # Инициализация HMC ядра
        kernel = HMC(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Инициализация MCMC
    mcmc = MCMC(kernel, num_samples=NUM_SAMPLES, warmup_steps=WARMUP_STEPS)
    return mcmc


# Вывод списка доступных методов с номером
print("Выберите метод сэмплинга:")
print("1. NUTS (No-U-Turn Sampler)")
print("2. HMC (Hamiltonian Monte Carlo)")



while True:
    chosen_number = input("Введите номер метода (1 или 2): ").strip()
    if chosen_number == "1":
        chosen_method = "NUTS"
        break
    elif chosen_number == "2":
        chosen_method = "HMC"
        break
    else:
        print("Неверный выбор! Пожалуйста, выберите 1 для NUTS или 2 для HMC.")


# Запуск MCMC с выбранным методом
mcmc = run_mcmc(chosen_method)
mcmc.run(data)

# Получение выборок для 'beta'
beta_samples: Tensor = mcmc.get_samples()['beta']

# Среднее по выборкам для 'beta'
mean_beta: Tensor = beta_samples.mean(0)
print("Estimated coefficients:", mean_beta)

# Построение гистограмм для каждого коэффициента
fig, axes = plt.subplots(1, DIM, figsize=(12, 4))

for i in range(DIM):
    axes[i].hist(beta_samples[:, i].detach().numpy(), bins=30, alpha=0.7, edgecolor='k')
    axes[i].set_title(f'Posterior distribution of beta_{i + 1}')
    axes[i].set_xlabel(f'beta_{i + 1}')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
