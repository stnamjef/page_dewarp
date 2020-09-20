# Page dewarp
- Faster version of [page_dewarp](https://github.com/mzucker/page_dewarp) in C++.
- [A dockerized test environment](https://hub.docker.com/repository/docker/stnamjef/opencv-4.0.0) is available.

## 1. Requirements

- Opencv 4.0 or greater.

- g++ 9.3 or greater.

## 2. Compile

- Pull the docker image 

```bash
# host shell
docker pull stnamjef/opencv-4.0.0:1.0
```

- Run the docker image

```bash
# pwd -> a host directory containing all the sources(.h, .cpp, images)
docker run -it -v $(pwd):/source stnamjef/opencv-4.0.0:1.0
```

- Compile

```bash
# container shell
g++ -o page_dewarp main.cpp $(pkg-config --cflags --libs opencv4)
```

## 3. Run

- Run page_dewarp

```bash
# container shell
./page_dewarp IMAGE1 IMAGE2 ...
```

## 4. Examples

- original image

![boston_cooking_a](./test images/boston_cooking_a.jpg)

- output

![boston_cooking_a.jpg_thresh](./test images/boston_cooking_a.jpg_thresh.png)