#pragma once

void startTimer();
float stopTimer();

void matrixMultiplicationGPUWrapper(float* A, float *B, float *C, int m, int n, int k, int version);