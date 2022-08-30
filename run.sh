#!/usr/bin/env bash

docker network create water

cd water-quality-backend
docker build -t water-quality-backend .
docker run --network=water -d -p 8888:8888 --name water-quality-backend water-quality-backend


cd ../water-quality-frontend
docker build -t water-quality-frontend .
docker run  --network=water -d -p 3000:6000 --name water-quality-frontend water-quality-frontend
