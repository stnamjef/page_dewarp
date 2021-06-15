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
# pwd -> a host directory containing all the sources(.h, .cpp)
docker run -it -v $(pwd):/source stnamjef/opencv-4.0.0:1.0
```

- Compile

```bash
# container shell
g++ -o page_dewarp main.cpp -std=c++17 -pthread $(pkg-config --cflags --libs opencv4)
```

## 3. Run

- Run page_dewarp.
- Warning: an input directory(-idir) must be given. Please check the CLI options below.

```bash
# container shell
./page_dewarp -idir=./test_images
```

## 4. Example

![example](C:\Users\stnam\Desktop\source\page_dewarp_master\page_dewarp\example.png)

## 5. Notice

- Supports parallel processing.
- CLI options(contributed by [zvezdochiot](https://github.com/ImageProcessing-ElectronicPublications/pagedewarp)) added as below.

| Options |  dtype  | description                                                  |
| :-----: | :-----: | ------------------------------------------------------------ |
|  -idir  | string  | Input directory containing images to process (default=NONE). |
|  -odir  | string  | Output directory where dewarped images should be saved (default=./dewarped). |
|   -nw   | integer | The number of workers(threads) for parallel processing (default=1). It is not recommended to set this option greater than the number of CPU cores. |
|   -mx   | integer | Reduced pixel to ignore near L/R edge of an image(default=50). |
|   -my   | integer | Reduced pixel to ignore near T/B edge of an image(default=20). |
|   -m    | integer | Reduced pixel to ignore near L/R/T/B edge of an image.       |
|   -z    |  float  | How much to zoom output relative to the original image(default=1.0). |
|   -d    | integer | Just affects stated DPI of PNG, not appearance(default=300). |
|   -r    | integer | Downscaling factor for remapping an image(default=16).       |
|   -tw   | integer | Min reduced pixel width of detected text contour(default=15). |
|   -th   | integer | Min reduced pixel height of detected text contour(default=2). |
|   -t    | integer | Min reduced pixel width and height of detected text contour. |
|   -ta   |  float  | Min text w/h ratio; Text contours below this value will be filtered out(default=1.5). |
|   -tt   | integer | Max reduced pixel thickness of detected text contour(default=10). |
|   -eo   |  float  | Max reduced pixel horizontal overlap of contours in span(default=1.0). |
|   -el   |  float  | Max reduced pixel length of edge connecting contours(default=100.0). |
|   -ea   |  float  | Cost of an angles in edges(default=10.0).                    |
|   -em   |  float  | Max change in angle allowed between contours(default=7.5).   |
|   -sw   | integer | Min reduced pixel width for span(default=30).                |
|   -sp   | integer | Reduced pixel spacing for sampling along spans(default=20).  |
|   -fl   |  float  | Normalized focal length of camera(default=1.2).              |
|   -ro   |         | Remap only (no threshold).                                   |
|   -db   |         | Debug mode (default=0).                                      |

