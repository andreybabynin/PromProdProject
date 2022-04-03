### This is a project for a PromProd course '22

#### Требования к файлам обучения

- файлы состоят из двух колонок ['type', 'text']
- варианты кодирования: ['utf-8', 'latin-1']
#### API

- GET /forward - получение предсказания по тексту письма 
    формат: curl -H "Content-Type: application/json" -d '{"text":"Hello World"}'

- POST /forward_batch
    формат: curl -H "Content-Type: multipart/form-data" -F file=@test.csv  

- GET /evaluate
    формат: curl -H "Content-Type: multipart/form-data" -F file=@test.csv
- /metadata
- /add_data - добавляет к основному датасету ранее загруженные файлы из /forward_batch
- /retrain - обучение на обновленном датасете с дефолтными параметрами
- /metrics
- /deploy