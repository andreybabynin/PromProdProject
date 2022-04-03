### This is a project for a PromProd course '22

GET /forward - получение предсказания по тексту письма 
    формат: curl -H "Content-Type: application/json" -d '{"text":"Hello World"}'

POST /forward_batch
    формат: curl -H "Content-Type: multipart/form-data" -F file=@test.csv  