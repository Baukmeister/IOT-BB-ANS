docker-compose build
docker push baukmeister/leader-app
docker push baukmeister/sensor-app

scp -r ./edgebench_scripts/leader/* jetson:leader-scripts
scp -r ./edgebench_scripts/sensors/* nuc:sensor-scripts
