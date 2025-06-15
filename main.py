import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS, HMC

NUM_SAMPLES: int = 500
WARMUP_STEPS: int = 300
DIM: int = 3  # Количество параметров x
BATCH_SIZE: int = 2000

Tensor = torch.Tensor  # Тип для тензоров

# Пример истинных значений параметров (могут быть любыми)
true_parameters: Tensor = torch.tensor([1., 2., 3.])


# Генерация случайных данных (например, измерения m)
# Можем взять случайные значения для m и известную зависимость F(x)
# Для F(x) выберем, например, линейную зависимость с некоторым шумом

def function(x: Tensor) -> Tensor:
    return x[0] + x[1] * x[2]  # Пример функции


# Генерация измерений m с ошибкой
m: Tensor = function(true_parameters) + torch.randn(BATCH_SIZE) * 0.1  # Добавляем шум


# Определение модели
def model(_m: Tensor) -> Tensor:
    # Начальные предположения для параметров x
    x_mean: Tensor = torch.zeros(DIM)  # Среднее для параметров
    x: Tensor = pyro.sample('x', dist.Normal(x_mean, torch.ones(DIM)))  # Параметры x с нормальным распределением

    # Ожидаемая ошибка между измеренным m и моделью F(x)
    predicted_m: Tensor = function(x)

    # Разница между m и F(x) с гауссовской ошибкой
    obs: Tensor = pyro.sample('obs', dist.Normal(predicted_m, 0.1), obs=_m)  # Наблюдения m с ошибкой
    return obs


# Функция для выполнения MCMC
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
    mcmc_out = MCMC(kernel, num_samples=NUM_SAMPLES, warmup_steps=WARMUP_STEPS)
    return mcmc_out


# Печать выбора метода
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

# Запуск MCMC
mcmc = run_mcmc(chosen_method)
mcmc.run(m)

# Извлечение сэмплов параметров x
x_samples: Tensor = mcmc.get_samples()['x']

# Среднее значение параметров
mean_x: Tensor = x_samples.mean(0)
print("Оцененные параметры:", mean_x)

# Построение гистограмм для каждого параметра
fig, axes = plt.subplots(1, DIM, figsize=(12, 4))

for i in range(DIM):
    axes[i].hist(x_samples[:, i].detach().numpy(), bins=30, alpha=0.7, edgecolor='k')
    axes[i].set_title(f'Posterior distribution of x_{i + 1}')
    axes[i].set_xlabel(f'x_{i + 1}')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()

# Построение графиков цепочек
fig, axes = plt.subplots(1, DIM, figsize=(12, 4))

for i in range(DIM):
    axes[i].plot(x_samples[:, i].detach().numpy(), alpha=0.7)
    axes[i].set_title(f'Chain for x_{i + 1}')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel(f'x_{i + 1}')

plt.tight_layout()
plt.show()
