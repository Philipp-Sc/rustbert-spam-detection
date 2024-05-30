FROM rust:latest

RUN apt-get update

RUN mkdir /usr/target

WORKDIR /usr/workspace

ENV CARGO_HOME=/usr/cargo_home
ENV CARGO_TARGET_DIR=/usr/target

CMD ["cargo build","cargo run"]
