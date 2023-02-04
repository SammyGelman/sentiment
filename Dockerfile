FROM ubuntu:latest
RUN apt-get update && apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh
RUN npm install
RUN npm install dockerode

FROM node:12

WORKDIR /app

COPY package.json /app

COPY . /app

EXPOSE 80

CMD ["which docker"]
# CMD ["node", "server.js"]
