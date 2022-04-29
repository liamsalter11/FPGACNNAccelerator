/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "cnn.h"


float IsError(float a, float b) {
	return fabs((a - b) / (a + b)) > 1e-3f && fabs(a - b) > 0.05f;
}

void cnn_sw( std::vector<DTYPE, aligned_allocator<DTYPE> > input, std::vector<DTYPE, aligned_allocator<DTYPE> > weight, std::vector<DTYPE, aligned_allocator<DTYPE> > & output){

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if( tid == 0 ){
            int nthreads = omp_get_num_threads();
            std::cout << "Running OpenMP with " << nthreads << " threads...\n";
        }
    }

	// Allocate memory on heap to avoid stack overflow.
    static float local_input[kInImSize][kInImSize][kNum];
    static float local_output[kOutImSize][kOutImSize][kNum];
    static float local_weight[kKernel][kKernel][kNum][kNum];
#pragma omp parallel for
    for (int h = 0; h < kInImSize; ++h) {
        for (int w = 0; w < kInImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_input[h][w][i] = input[(h*kInImSize+w)*kNum+i];
            }
        }
    }
#pragma omp parallel for
    for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q){
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    local_weight[p][q][i][j] = weight[((p*kKernel+q)*kNum+i)*kNum+j];
                }
            }
        }
    }
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                local_output[h][w][i] = 0.0f;
            }
        }
    }
	// Convolution
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w) {
            for (int i = 0; i < kNum; ++i) {
                for (int j = 0; j < kNum; ++j) {
                    for (int p = 0; p < kKernel; ++p) {
                        for (int q = 0; q < kKernel; ++q){
                            local_output[h][w][i] += local_input[h+p][w+q][j] * local_weight[p][q][i][j];
                        }
                    }
				}
			}
		}
	}
#pragma omp parallel for
    for (int h = 0; h < kOutImSize; ++h) {
        for (int w = 0; w < kOutImSize; ++w){
            for (int i = 0; i < kNum; ++i) {
                output[(h*kOutImSize + w)*kNum+i] = local_output[h][w][i];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string binaryFile = argv[1];

    cl_int err;
    cl::Context context;
    cl::Kernel krnl_cnn;
    cl::CommandQueue q;
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
    // hood user ptr
    // is used if it is properly aligned. when not aligned, runtime had no choice
    // but to create
    // its own host side buffer. So it is recommended to use this allocator if
    // user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
    // boundary. It will
    // ensure that user buffer is used when user create Buffer/Mem object with
    // CL_MEM_USE_HOST_PTR

    std::vector<DTYPE, aligned_allocator<DTYPE> > A(kNum*kInImSize*kInImSize); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > B(kNum*kNum*kKernel*kKernel); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > AB_sw(kNum*kOutImSize*kOutImSize); 
    std::vector<DTYPE, aligned_allocator<DTYPE> > AB_hw(kNum*kOutImSize*kOutImSize); 

	std::cout << "Initializing input data...\n";
    srand(time(NULL)); 
#pragma omp parallel for
	for (int i = 0; i < kNum*kOutImSize*kOutImSize; ++i) {
		AB_sw[i] = 0.0f;
		AB_hw[i] = 0.0f;
	}
#pragma omp parallel for
	for (int i = 0; i < kNum*kInImSize*kInImSize; ++i) {
		A[i] = -2 + static_cast <float>(rand()) /( static_cast <float> (RAND_MAX/(4)));
	}

#pragma omp parallel for
	for (int i = 0; i < kNum*kNum*kKernel*kKernel; ++i) {
		B[i] = -2 + static_cast <float>(rand()) /( static_cast <float> (RAND_MAX/(4)));
	}


    printf("Done initializing vectors\n");

    std::cout << "Running SW CNN...\n";
    auto start = std::chrono::steady_clock::now();
    cnn_sw(A, B, AB_sw);
    auto end = std::chrono::steady_clock::now();
    double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double gops = double(kKernel) * kKernel * kNum * kNum * kOutImSize * kOutImSize * 2 / (exec_time);
    std::cout << "CPU CNN Time: " << exec_time*1e-9 << " sec,CPU CNN GOPS: " << gops << std::endl;
    printf("Done\n");

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_cnn = cl::Kernel(program, "cnn", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DTYPE)*kNum*kInImSize*kInImSize, A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(DTYPE)*kNum*kNum*kKernel*kKernel, B.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_AB(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(DTYPE)*kNum*kOutImSize*kOutImSize, AB_hw.data(), &err));

    OCL_CHECK(err, err = krnl_cnn.setArg(0, buffer_A));
    OCL_CHECK(err, err = krnl_cnn.setArg(1, buffer_B));
    OCL_CHECK(err, err = krnl_cnn.setArg(2, buffer_AB));

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A, buffer_B}, 0 /* 0 means from host*/));
    q.finish();
    
    std::cout << "Running FPGA CNN...\n";

    start = std::chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_cnn));
    q.finish();

    end = std::chrono::steady_clock::now();
    std::cout << "Done.\n";
    exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    gops = double(kKernel) * kKernel * kNum * kNum * kOutImSize * kOutImSize * 2 / (exec_time);
    std::cout << "FPGA CNN Time: " << exec_time*1e-9 << " sec, FPGA GOPS: " << gops << std::endl;

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_AB}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();



	int error = 0;
	bool first = true;
    for (int h = 0; h < kOutImSize; ++h) {
            for (int w = 0; w < kOutImSize; ++w) {
                for (int i = 0; i < kNum; ++i) {
            //    std::cout << "Check Got " << AB_hw[(h*kOutImSize+w)*kOutImSize+i] << ", expecting "
			//				<< AB_sw[(h*kOutImSize+w)*kOutImSize+i] << ", h = " << h
			//				<< ", w = " << w << " @ i = " << i << std::endl;
				if (IsError(AB_hw[(i*kOutImSize+h)*kOutImSize+w], AB_sw[(i*kOutImSize+h)*kOutImSize+w])) {
					if (first) {
						std::cout << "First error: Got " << AB_hw[(i*kOutImSize+h)*kOutImSize+w] << ", expecting "
							<< AB_sw[(i*kOutImSize+h)*kOutImSize+w] << " @ i = " << i << ", h = " << h
							<< ", w = " << w << std::endl;
						first = false;
					}
					++error;
				}
			}
		}
	}
	if (error != 0) {
		std::cout << "Found " << error << " error" << (error > 1 ? "s\n" : "\n");
		std::cout << "FPGA CNN FAIL" << std::endl;
		return EXIT_FAILURE;
	} else {
		std::cout << "FPGA CNN PASS" << std::endl;
		return EXIT_SUCCESS;
	}
}

