#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <device_functions.h> 
#include "device_launch_parameters.h"
#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "bmp/EasyBMP.h"
#include <stdio.h>

using namespace std;

#define WINDOW_SIZE 3
#define COUNT_POINTS 9

#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
printf("Cuda error: %s\n", cudaGetErrorString(err));    \
printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
system("pause");\
}       


// объявляем ссылку на текстуру для двумерной текстуры float
texture<float, cudaTextureType2D, cudaReadModeElementType> tex;


__global__ void medianFilter(float *output, int imageWidth, int imageHeight) {
	//  выбрали строку и столбец для потока
	int col = blockIdx.x *  blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	// маска по которой будет находиться медиана
	float mask[COUNT_POINTS] = { 0,0,0,0,0,0,0,0,0 };
	
	int k = 0;
	// Т.к текстуры обладают свойством свертывание - т.е выход за границы, будем идти от -1 до 1 с шагом 1 по картинки, 
	//заполняя маску  
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			mask[k] = tex2D(tex, col + j, row + i);
			k++;
		}
	}

	// Отсортировали значения в маске 
	for (int i = 1; i < COUNT_POINTS; i++) {
		for (int j = i; j > 0 && mask[j - 1] > mask[j]; j--) {
			int tmp = mask[j - 1];
			mask[j - 1] = mask[j];
			mask[j] = tmp;
		}
	}
	// в результирующий центральный пиксель записали медиану
	output[row * imageWidth + col] = mask[4];
	
}


float *readImage(char *filePathInput, unsigned int *rows, unsigned int *cols) {
	BMP Image;
	Image.ReadFromFile(filePathInput);
	*rows = Image.TellHeight();
	*cols = Image.TellWidth();
	float *imageAsArray = (float *)calloc(*rows * *cols, sizeof(float));
	// Преобразуем картику в черно-белую
	for (int i = 0; i < Image.TellWidth(); i++)	{
		for (int j = 0; j < Image.TellHeight(); j++) {
			double Temp = 0.30*(Image(i, j)->Red) +	0.59*(Image(i, j)->Green) +	0.11*(Image(i, j)->Blue);
			Image(i, j)->Red = (unsigned char)Temp;
			Image(i, j)->Green = (unsigned char)Temp;
			Image(i, j)->Blue = (unsigned char)Temp;
			imageAsArray[j * *cols + i] = Temp;
		}
	}
	return imageAsArray;
}


void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols) {
	BMP Output;
	Output.SetSize(cols, rows);
	// записали картинку 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			RGBApixel pixel;
			pixel.Red = grayscale[i * cols + j];
			pixel.Green = grayscale[i * cols + j];
			pixel.Blue = grayscale[i * cols + j];
			pixel.Alpha = 0;
			Output.SetPixel(j, i, pixel);
		}
	}
	Output.WriteToFile(filePath);
}

int main() {
	setlocale(LC_ALL, "RUS");

	unsigned int rows, cols;
	// считали картинку 
	float * imageAsArray = readImage ("lena.bmp", &rows, &cols);

	//Создали дескриптор канала с форматом Float
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray *cuArray;

	// Выделили Cuda массив соответствии со структурой cudaChannelFormatDesc и вернули дескриптор нового массива CUDA в cuArray
	CUDA_CHECK_ERROR(cudaMallocArray(&cuArray, &channelDesc, cols, rows));
	// Скопировали массив imageAsArray в cuArray
	CUDA_CHECK_ERROR(cudaMemcpyToArray(cuArray, 0, 0, imageAsArray, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

	// Установили параметры текстуры
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModeLinear;

	// Привязали массив к текстуре
	CUDA_CHECK_ERROR(cudaBindTextureToArray(tex, cuArray, channelDesc));

	float *dev_output, *output;
	float gpuTime = 0;

	output = (float *)calloc(rows * cols, sizeof(float));

	CUDA_CHECK_ERROR(cudaMalloc(&dev_output, rows * cols * sizeof(float)));

	dim3 dimBlock(16, 16);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,
		(rows + dimBlock.y - 1) / dimBlock.y);

	cudaEvent_t start;
	cudaEvent_t stop;

	//Создаем event'ы для синхронизации и замера времени работы GPU
	CUDA_CHECK_ERROR(cudaEventCreate(&start));
	CUDA_CHECK_ERROR(cudaEventCreate(&stop));
	//Отмечаем старт расчетов на GPU
	cudaEventRecord(start, 0);

	medianFilter << <dimGrid, dimBlock >> > (dev_output, cols, rows);

	//Копируем результат с девайса на хост в output
	CUDA_CHECK_ERROR(cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

	//Отмечаем окончание расчета
	cudaEventRecord(stop, 0);

	//Синхронизируемя с моментом окончания расчетов
	cudaEventSynchronize(stop);

	//Рассчитываем время работы GPU
	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "Время на GPU = " << gpuTime << " мсек" << std::endl;
	writeImage("result.bmp", output, rows, cols);

	//Чистим ресурсы на видеокарте
	CUDA_CHECK_ERROR(cudaFreeArray(cuArray));
	CUDA_CHECK_ERROR(cudaFree(dev_output));

	system("pause");
	return 0;
}
