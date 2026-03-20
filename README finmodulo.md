# Dockerizar aplicación Dash Financiero

## Construir la imagen
docker build -t dash_finmodulo .

## Ejecutar el contenedor
docker run -p 9000:9000 dash_finmodulo 
