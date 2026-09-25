FROM golang:1.22-alpine AS build
WORKDIR /src
COPY minio-source.tar.gz /tmp/minio-source.tar.gz
RUN tar -xzf /tmp/minio-source.tar.gz --strip-components=1 -C /src
ENV CGO_ENABLED=0
ENV GOPROXY=https://goproxy.cn|https://proxy.golang.org|direct
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -ldflags="-s -w" -o /out/minio .

FROM alpine:3.20
RUN apk add --no-cache ca-certificates curl
COPY --from=build /out/minio /usr/local/bin/minio
