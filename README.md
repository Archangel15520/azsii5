# Практика 5: Атака с ограниченной памятью (PGD - Projected Gradient Descent)

# Студент: Васильев Григорий Максимович

# Группа: ББМО-02-23

# Цель задания: 

Изучить одну из наиболее мощных атак на модели ИИ — атаку Projected Gradient Descent (PGD). Научиться использовать PGD для создания противоречивых примеров и оценить её влияние на обученные модели.

# Задачи:

* Загрузить ранее обученную модель на датасете MNIST.
* Изучить теоретические основы атаки PGD.
* Реализовать атаку PGD с помощью фреймворка Foolbox.
* Оценить точность модели на противоречивых примерах и сравнить с результатами на обычных данных.

  ---

**WARNING(Важная информация): 1. Все работы по данному предмету можно найти по ссылке: https://github.com/Archangel15520/AZSII-REPO/tree/main**

**2. В коде используется ранее обученная модель на датасете MNIST, которую можно найти в закрепе к данному проекту.**

**3. Сылка на выполненую работу в среде google colab: https://colab.research.google.com/drive/1yqpxrr3Khnq-dNKFPJSt1d6O2IDtxh-A#scrollTo=BUoPd6nalqFf** 

  ---
  
# Загрузка обученной модели и данных MNIST

```
# Установка необходимой библиотеки:
!pip install foolbox

# Загрузка обученной модели и данных MNIST
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка тестовых данных MNIST
(_, _), (test_images, test_labels) = mnist.load_data()

# Нормализация данных
test_images = test_images / 255.0

# Преобразование меток в формат one-hot
test_labels = to_categorical(test_labels, num_classes=10)

# Загрузка обученной модели
model = tf.keras.models.load_model('mnist_model.h5')

# Проверка точности модели на обычных данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy on clean images: {test_acc}')
```

![image](https://github.com/Archangel15520/azsii5/blob/main/screenshot/1.JPG)

# Реализация атаки PGD с использованием Foolbox

```
import torch
import foolbox as fb
import numpy as np
import matplotlib.pyplot as plt

# Инициализация Foolbox модели
fmodel = fb.TensorFlowModel(model, bounds=(0, 1))

# Выбор изображения для атаки (например, первое изображение из тестового набора)
image = tf.convert_to_tensor(test_images[0], dtype=tf.float32)[None, ...]
label = np.argmax(test_labels[0])
label = tf.convert_to_tensor(label, dtype=tf.int64)

# Выполнение атаки
attack = fb.attacks.LinfPGD()
advs, _, success = attack(fmodel, image, label[None], epsilons=0.1)

# Вывод результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Оригинальное изображение")
plt.imshow(image[0].numpy(), cmap="gray")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Противоречивое изображение (PGD)")
plt.imshow(advs[0].numpy(), cmap="gray")
plt.axis('off')

plt.show()
```

![image](https://github.com/Archangel15520/azsii5/blob/main/screenshot/2.JPG)

# Оценка модели на противоречивых примерах

```
adversarial_images = []

# Обработка изображений
for i in range(len(test_images)):
    image = tf.convert_to_tensor(test_images[i], dtype=tf.float32)[None, ...]
    label = np.argmax(test_labels[i])
    label = tf.convert_to_tensor(label, dtype=tf.int64)
    advs, _, success = attack(fmodel, image, label[None], epsilons=0.1)
    adversarial_images.append(advs)

adversarial_images = tf.concat(adversarial_images, axis=0)

adversarial_loss, adversarial_acc = model.evaluate(adversarial_images, test_labels)
print(f'Accuracy on adversarial examples (PGD): {adversarial_acc}')
```

![image](https://github.com/Archangel15520/azsii5/blob/main/screenshot/3.JPG)

# Вывод:

Точность модели снизилась с 97.8% на чистых тестовых изображениях до 5.1% на атакованных примерах, что демонстрирует её уязвимость перед атакой PGD. Эта атака, минимально изменяя входные данные, существенно снижает производительность модели, подчёркивая необходимость разработки устойчивых методов машинного обучения.

Для повышения устойчивости модели можно использовать стратегии, такие как аугментация данных с противоречивыми примерами, регуляризация или adversarial training. Также важным шагом является внедрение устойчивых архитектур, способных ограничивать влияние атак, и систематическое тестирование модели на различные виды атак для повышения её надёжности, особенно в критически важных приложениях, где ошибки недопустимы.
