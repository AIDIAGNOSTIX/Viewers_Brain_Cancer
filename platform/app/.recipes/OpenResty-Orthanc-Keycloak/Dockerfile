# Stage 1: Build the application
FROM node:18.16.1-slim as builder

RUN mkdir /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y build-essential python3

ENV APP_CONFIG=config/docker_openresty-orthanc-keycloak.js
ENV PATH /usr/src/app/node_modules/.bin:$PATH

# Copy all files from the root of the OHIF source
COPY ./ /usr/src/app/

# Install dependencies and build the application
RUN yarn config set workspaces-experimental true
RUN yarn install
RUN yarn run build

# Verify that the build output exists
RUN ls /usr/src/app/platform/app/dist

# Stage 2: Bundle the built application into a Docker container
FROM openresty/openresty:1.21.4.2-0-bullseye-fat

# Copy SSL certificates and Nginx configuration
COPY ./platform/app/.recipes/OpenResty-Orthanc-Keycloak/config/ssl /usr/local/openresty/nginx/conf/ssl
COPY ./platform/app/.recipes/OpenResty-Orthanc-Keycloak/config/nginx.conf /usr/local/openresty/nginx/conf/nginx.conf

RUN mkdir /var/log/nginx
RUN apt-get update && \
  apt-get install -y openssl libssl-dev git gcc wget unzip make && \
  apt-get clean

RUN apt-get install --assume-yes lua5.4 libzmq3-dev lua5.4-dev
RUN cd /tmp && \
  wget http://luarocks.org/releases/luarocks-3.9.2.tar.gz && \
  tar zxpf luarocks-3.9.2.tar.gz && \
  cd luarocks-3.9.2 && \
  ./configure && \
  make && \
  make install

RUN luarocks install lua-resty-http
RUN luarocks install lua-cjson
RUN luarocks install lua-resty-string
RUN luarocks install lua-resty-session
RUN luarocks install lua-resty-jwt
RUN luarocks install lua-resty-openidc
RUN luarocks install lua-resty-auto-ssl

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy build output to image
COPY --from=builder /usr/src/app/platform/app/dist /var/www/html

# Verify that the files have been copied
RUN ls /var/www/html

ENTRYPOINT ["/usr/local/openresty/nginx/sbin/nginx", "-g", "daemon off;"]
