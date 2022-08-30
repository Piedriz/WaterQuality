FROM node:16-alpine3.14 as build
ARG VITE_API_URL
WORKDIR /app
COPY package.json .
COPY yarn.lock .
RUN yarn
COPY . .
CMD [ "yarn", "dev" ]
