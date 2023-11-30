# ML_car_price_pred

Выполнил: Василенко П.О.

Что было сделано:  
  1) Были обработаны признаки тренировочного и тестового датасетов: посчитаны основные статистики числовых и категориальных столбцов, найдены пропуски и заменены на медианы признаков, найдены повторяющиеся объекты и удалены из датасетов, убраны единицы измерения для признаков mileage, engine, max_power и приведены к типу float, признак torque разбит на max_torque_rpm и torque признаки (единицы измерения учтены), engine и seats преобразованы к int.  
  2) Были построены попарные распределения числовых признаков для train и test, кратко описаны связи предикторов с целевой переменной и корреляции признаков. Для трейна построена тепловая карта попарных корреляций числовых признаков и на ее основе о наименее и наиболее скоррелированной паре признаков (см. ноутбук), а также отображена диаграмма рассеяния наиболее скоррелированной пары признаков.  
  3) Были обучены на вещественных признаках модели: LinearRegression, Lasso, ElasticNet. После LinearRegression использовался StandardScaler для признаков. Для оценки качества моделей были выведены $R^2$ и MSE.  
  4) Добавлены категориальные признаки (учитвая seats) и закодированы методом OneHotEncoder. Обучена модель Ridge с перебором параметров через GridSearchCV. Для оценки качества модели были выведены $R^2$ и MSE.  
  5) В Feature Engineering проверены признаки на выброс (один выброс был исключен) и проверена целевая переменная. Из неиспользованного признака name в исходном датасете был получен новый признак model для дальнейшего обучения. Сгенерированы полиномиальные признаки 3 степени для улучшения качества модели Ridge.  
  6) Рассчитана кастомная метрика для лучшей из моделей.  
  7) Наконец, реализован сервис на FastAPI, позволяющий при подаче объекта заданного класа (список признаков машины) или коллекции объектов получать предсказанную стоимость машины:
   - на вход в формате json подаются признаки одного объекта, на выходе сервис выдает предсказанную стоимость машины;
   - на вход подается csv-файл с признаками тестовых объектов, на выходе получаем файл с +1 столбцом - предсказаниями на этих объектах.
      
С какими результатами:  
 
  После успешной обработки датафреймов, на числовых данных были обучены модели, для которых лучший показатель качества модели $R^2$ составлял ~60%. После добваления категориальных признаков $R^2$ увеличился на ~6.5%. С применением методов, описанных в Feature Engineering, показатель возрос до ~93%. По итогу, $R^2$ с 60% увеличился до 93%.
  
Что дало наибольший буст в качестве:  
  
  Наибольший буст в качестве дало создание нового признака model и полиномиальных признаков (суммарно ~17%).
  
Что сделать не вышло и почему (это нормально, даже хорошо😀):  
  
  Не хватило времени для более глубокого анализа корреляции признаков между собой с целью удаления тех, которые сильно скоррелированы между собой (например, max_power и engine сильно скоррелированы между собой, вероятно, удаление одного из них несильно бы поменяло коэффициент качества). Можно было бы посмотреть и проверить другие методы Feature Engineering, описанные в задании, но не успел)).  
  ***Хотелось бы отметить, что на задание с разделением признака torque ушло достаточно много времени и сил, а оценивается оно в два раза меньше по баллам, чем задание с бизнесовой метрикой, в которой пара строчек кода. Может быть, стоит увеличить кол-во баллов за это задание))*** :smile:

Запуск локального сервера:
![image](https://github.com/Pixel-Pirate-Coder/ML_car_price_pred/assets/145439150/df2afef6-e944-4fa8-ac7b-785b3dd9033c)

Для одного объекта до запроса в postman:
![image](https://github.com/Pixel-Pirate-Coder/ML_car_price_pred/assets/145439150/fb44d355-e280-4384-8823-102eb70ed2a0)

После запроса в postman:
![image](https://github.com/Pixel-Pirate-Coder/ML_car_price_pred/assets/145439150/e3d68899-eb97-421f-bf64-f3df64c7dce2)

Для коллекции объектов до запроса в postman:
![image](https://github.com/Pixel-Pirate-Coder/ML_car_price_pred/assets/145439150/ffdfe8db-dd61-4fb1-bae7-ac049a57f6d3)

После запроса в postman:
![image](https://github.com/Pixel-Pirate-Coder/ML_car_price_pred/assets/145439150/da87b1c2-61ef-45b1-82b8-4616e8d55aca)




