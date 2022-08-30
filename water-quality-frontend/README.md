# Water Quality

## Start with docker:

- Build the image

```sh
docker build --build-arg VITE_API_URL=http://127.0.0.1:8888 -t frontend-water-quality .
```

- Run the container

```sh
docker run --rm -p 3000:6000 frontend-water-quality
```
