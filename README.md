# MedianFilter

# About app
The Median Filter is a non-linear digital filtering technique, often used to remove noise from an image or signal. 
Such noise reduction is a typical pre-processing step to improve the results of later processing 
(for example, edge detection on an image).

## Using
1. Install CUDA on your computer (required to have a video card from Nvidia).
2. Create new CUDA project in Visual Studio.
3. Copy this code.
4. Run the application.


## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | AMD Ryzen 5 2600 Six-Core Processor 3.4 GHz (Turbo Boost 3.9 GHz) |
| RAM  | 16 GB DDR4 |
| GPU  | GIGABYTE GeForce GTX 550 Ti [GV-N550D5-1GI]  |
| OS   | Windows 10 64-bit  |


## Results
| size image | time CPU |  time GPU  | Acceleration | 
|------------|----------|------------|--------------|
|   64х64    | 4 ms     |0.35 ms     |    11.42     |  
|   128х128  | 10 ms    |0.58 ms     |    17.24     | 
|   256х256  | 38 ms    | 2.06 ms    |    18.45     |   
|   512х512  | 173 ms   | 8.9 ms     |    19.4      |   
|   1024х1024| 591 ms   | 26.1 ms    |    22.64     |   
|   2048х2048| 2425 ms  | 92.3 ms    |    26.27     | 

## Input Image
![Описание изображения](lena.bmp)

## OutputCPU Image
![Описание изображения](resultCPU.bmp)

## OutputGPU Image
![Описание изображения](resultGPU.bmp)
