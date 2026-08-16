#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include <cuda_runtime.h>

constexpr int THREADS_PER_SIDE = 8;
constexpr int THREADS_PER_BLOCK = THREADS_PER_SIDE * THREADS_PER_SIDE * THREADS_PER_SIDE;
constexpr int PAD = 2;

// Encapsulate grid infos in a single structure
struct Grid {
    int x_n, y_n, z_n;
    float dx, dy, dz, dt;
    float a_w, a_m;

    Grid(int x_n, int y_n, int z_n, float dx, float dy, float dz, float dt, float a_w, float a_m) : 
        x_n(x_n), y_n(y_n), z_n(z_n),
        dx(dx), dy(dy), dz(dz), dt(dt),
        a_w(a_w), a_m(a_m) {}
};

// Simulation loop without shared memory
__global__ void naive_simulation_loop(const float* T_curr, float* T_next, const int* h_pos, const Grid grid) {
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    int z_id = blockIdx.z * blockDim.z + threadIdx.z;

    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const float dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const float a_w = grid.a_w, a_m = grid.a_m;
    
    if ((y_id > 0 && y_id < y_n-1) && (x_id > 0 && x_id < x_n-1) && (z_id > 0 && z_id < z_n-1)) {
        int t_idx = x_id*y_n*z_n + y_id*z_n + z_id;
        float laplacian = (
            (T_curr[t_idx + y_n*z_n] - 2*T_curr[t_idx] + T_curr[t_idx - y_n*z_n])/dx2 + 
            (T_curr[t_idx + z_n]     - 2*T_curr[t_idx] + T_curr[t_idx - z_n])/(dy2) + 
            (T_curr[t_idx + 1]       - 2*T_curr[t_idx] + T_curr[t_idx - 1])/(dz2) 
        );

        // Use correct thermal diffusivity depending on position
        float a;
        if ((x_id>=h_pos[0] && x_id<h_pos[1]) && 
            (y_id>=h_pos[2] && y_id<h_pos[3]) &&
            (z_id>=h_pos[4] && z_id<h_pos[5])) 
            a = a_m; 
        else
            a = a_w;

        // Update rule
        T_next[t_idx] = T_curr[t_idx] + a*dt*laplacian;
    }
}

// Simulation loop with shared memory
__global__ void optimal_loop_simulation(const float* T_curr, float* T_next, const int* h_pos, const Grid grid) {
    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const float dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const float a_w = grid.a_w, a_m = grid.a_m;

    int tid_x = threadIdx.x, bdim_x = blockDim.x;
    int tid_y = threadIdx.y, bdim_y = blockDim.y;
    int tid_z = threadIdx.z, bdim_z = blockDim.z;

    int x_id = bdim_x * blockIdx.x + tid_x;
    int y_id = bdim_y * blockIdx.y + tid_y;
    int z_id = bdim_z * blockIdx.z + tid_z;

    // Shared (memory) index
    int sid_x = tid_x + PAD/2;
    int sid_y = tid_y + PAD/2;
    int sid_z = tid_z + PAD/2;

    // Global index
    auto gid = [&](int xid, int yid, int zid) {
        return xid * y_n * z_n + yid * z_n + zid;
    };

    // Shared memory
    __shared__ float stencil_T[THREADS_PER_SIDE+PAD][THREADS_PER_SIDE+PAD][THREADS_PER_SIDE+PAD];

    // Loading "bulk" of the shared memory
    stencil_T[sid_x][sid_y][sid_z] = T_curr[gid(x_id, y_id, z_id)];
    
    // Loading surface boundaries (top, down, left, right, front, back)
    // Note: linear and puntual boundaries are not needed for the simulation!
    if (tid_z < PAD/2 && z_id-PAD/2 >= 0) {
        stencil_T[sid_x][sid_y][tid_z] = T_curr[gid(x_id, y_id, z_id-PAD/2)];
    }
    if (tid_z >= bdim_z-PAD/2 && z_id+PAD/2 < z_n) {
        stencil_T[sid_x][sid_y][sid_z+PAD/2] = T_curr[gid(x_id, y_id, z_id+PAD/2)];
    }
    if (tid_y < PAD/2 && y_id-PAD/2 >= 0) {
        stencil_T[sid_x][tid_y][sid_z] = T_curr[gid(x_id, y_id-PAD/2, z_id)];
    }
    if (tid_y >= bdim_y-PAD/2 && y_id+PAD/2 < y_n) {
        stencil_T[sid_x][sid_y+PAD/2][sid_z] = T_curr[gid(x_id, y_id+PAD/2, z_id)];
    }
    if (tid_x < PAD/2 && x_id-PAD/2 >= 0) {
        stencil_T[tid_x][sid_y][sid_z] = T_curr[gid(x_id-PAD/2, y_id, z_id)];
    }
    if (tid_x >= bdim_x-PAD/2 && x_id+PAD/2 < x_n) {
        stencil_T[sid_x+PAD/2][sid_y][sid_z] = T_curr[gid(x_id+PAD/2, y_id, z_id)];
    }
    __syncthreads();
    
    if ((y_id > 0 && y_id < y_n-1) && (x_id > 0 && x_id < x_n-1) && (z_id > 0 && z_id < z_n-1)) {
        float laplacian = (
            (stencil_T[sid_x+1][sid_y][sid_z] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x-1][sid_y][sid_z])/dx2 + 
            (stencil_T[sid_x][sid_y+1][sid_z] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x][sid_y-1][sid_z])/dy2 + 
            (stencil_T[sid_x][sid_y][sid_z+1] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x][sid_y][sid_z-1])/dz2 
        );

        // Use correct thermal diffusivity depending on position
        float a;
        if ((x_id>=h_pos[0] && x_id<h_pos[1]) && 
            (y_id>=h_pos[2] && y_id<h_pos[3]) &&
            (z_id>=h_pos[4] && z_id<h_pos[5])) 
            a = a_m; 
        else
            a = a_w;

        // Update rule
        T_next[gid(x_id, y_id, z_id)] = stencil_T[sid_x][sid_y][sid_z] + a*dt*laplacian;
    }
}

__global__ void compute_temperatures(const float* T, float2* R, const Grid grid, const int* h_pos) {
    int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;

    // Heater position
    int h_xmin = h_pos[0], h_xmax = h_pos[1];
    int h_ymin = h_pos[2], h_ymax = h_pos[3];
    int h_zmin = h_pos[4], h_zmax = h_pos[5];

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int start_x = 2 * blockDim.x * blockIdx.x;
    int start_y = 2 * blockDim.y * blockIdx.y;
    int start_z = 2 * blockDim.z * blockIdx.z;

    // Mapping unique (3D-)thread index onto shared memory 
    int s_tid = tid_x * blockDim.y * blockDim.z + tid_y * blockDim.z + tid_z; 
    __shared__ float2 partial_sum[THREADS_PER_BLOCK];

    float T_in = 0.;
    float T_out = 0.;

    // Compilator unrolls nested loops into sequential assembly code
    #pragma unroll
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            for (int k=0; k<2; ++k) {
                
                int t_tid_x = start_x + tid_x + (i*blockDim.x);
                int t_tid_y = start_y + tid_y + (j*blockDim.y);
                int t_tid_z = start_z + tid_z + (k*blockDim.z);

                if ((t_tid_x < x_n) && (t_tid_y < y_n) && (t_tid_z < z_n)) {
                    float val = T[t_tid_x*y_n*z_n + t_tid_y*z_n + t_tid_z];

                    bool inside = (t_tid_x >= h_xmin && t_tid_x < h_xmax) &&
                                  (t_tid_y >= h_ymin && t_tid_y < h_ymax) && 
                                  (t_tid_z >= h_zmin && t_tid_z < h_zmax);
                    
                    if (inside) T_in += val;
                    else        T_out += val;
                }
            }
        }
    }

    // Load shared memory with partial sums
    partial_sum[s_tid] = make_float2(T_out, T_in);
    __syncthreads();

    // Performing reduction
    for (int stride = 2; stride <= THREADS_PER_BLOCK; stride *= 2) {
        if (s_tid < THREADS_PER_BLOCK/stride) {
            partial_sum[s_tid].x += partial_sum[s_tid + THREADS_PER_BLOCK/stride].x;
            partial_sum[s_tid].y += partial_sum[s_tid + THREADS_PER_BLOCK/stride].y;
        }
        __syncthreads();
    }

    if (s_tid == 0) {
        R[blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z] = partial_sum[0];
    }
}

int main(int argc, char* argv[]) {
    // -------------- List of flags that can be turned on from terminal --------------

    bool save_data = false;        // save in a file temperature in function of time
    bool show_verbosity = false;   // show execution on terminal
    bool use_tiling = false;       // use reduction directive (to implement different ones)

    // -------------------------------------------------------------------------------
    // Check arguments
    if (argc > 1) {
        for (int i=1; i<argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--save-data") {
                save_data = true;
            } else if (arg == "--show-verbosity") {
                show_verbosity = true;
            } else if (arg == "--use-tiling") {
                use_tiling = true;
            } else {
                printf("Error! argumnt (%d) is not implemented!", i);
                return 1;
            }
        }
    }


    // Grid specifications
    int t_n = 2000;
    int x_n = 100, y_n = 100, z_n = 100;
    int tol = 10;

    // Systems physical dimensions (space: mm, time: sec)
    const float x_len = 10., y_len = 10., z_len = 10.;
    const float xx_len = 2, yy_len = 2, zz_len = 2;
    const float a_w = 0.143, a_c = 111;

    // Position of internal heater
    const float xx_pos = (x_len-xx_len)/2;
    const float yy_pos = (y_len-yy_len)/2;
    const float zz_pos = (z_len-zz_len)/2;

    // We add a heater at temperature T
    float T = 300, T0 = 280;
    float dx = x_len/x_n, dy = y_len/y_n, dz = z_len/z_n, dt = 1e-6;  
    printf("\n------------------------------------------------------\n");
    printf("Running heat transfer simulation with:\n");
    printf("• Points along (x,y,z): %d, %d, %d (total: %d)\n", x_n, y_n, z_n, x_n*y_n*z_n);
    printf("• Step-size along (x,y,z,t): %1.3f, %1.3f, %1.3f, %1.3e\n", dx, dy, dz, dt);
    printf("• Duration of the simulation: %1.5e\n", dt*t_n);

    // Veryfing we have enough points    
    if (xx_len/dx<tol) {
        printf("Error: not enough points (%d) along x in the small system.", (int)(xx_len/dx));
    } else if (yy_len/dy<tol) {
        printf("Error: not enough points (%d) along y in the small system.", (int)(yy_len/dy));
    } else if (zz_len/dz<tol) {
        printf("Error: not enough points (%d) along z in the small system.", (int)(z_len/dz));
    }

    // Indices for heater position
    int xx_idx_start = xx_pos/dx, xx_idx_end = (xx_pos+xx_len)/dx;
    int yy_idx_start = yy_pos/dy, yy_idx_end = (yy_pos+yy_len)/dy;
    int zz_idx_start = zz_pos/dz, zz_idx_end = (zz_pos+zz_len)/dz;
    int p_in = (xx_idx_end-xx_idx_start)*(yy_idx_end-yy_idx_start)*(zz_idx_end-zz_idx_start);

    printf("• Points along (x,y,z) in the internal system: %d, %d, %d (total: %d)\n", xx_idx_end-xx_idx_start, yy_idx_end-yy_idx_start, zz_idx_end-zz_idx_start, p_in);
    
    // Setting block size and grid size
    int dim = x_n*y_n*z_n;
    dim3 blockSize(THREADS_PER_SIDE, THREADS_PER_SIDE, THREADS_PER_SIDE);
    auto calc_grid = [&](int n) {
        return dim3(
            (x_n + n*blockSize.x - 1)/(n*blockSize.x),
            (y_n + n*blockSize.y - 1)/(n*blockSize.y),
            (z_n + n*blockSize.z - 1)/(n*blockSize.z)
        );
    };

    // Used for computing temperatures
    dim3 tempGridSize = calc_grid(2);

    // Used for running simulations
    dim3 simulGridSize = calc_grid(1);
    int n_blocks = tempGridSize.x * tempGridSize.y * tempGridSize.z;

    std::vector<float> h_T_curr(dim, T0);
    std::vector<float> h_T_next(dim, T0);
    std::vector<float2> h_R(n_blocks, make_float2(0., 0.));

    // Parameters containing simulation grid's informations
    Grid grid(x_n, y_n, z_n, dx, dy, dz, dt, a_w, a_c);
    int h_heater_id[6] = {
        xx_idx_start, xx_idx_end,
        yy_idx_start, yy_idx_end,
        zz_idx_start, zz_idx_end
    };

    // Setting initial temperature of the internal heater
    for (int i=xx_idx_start; i<xx_idx_end; ++i) {
        for (int j=yy_idx_start; j<yy_idx_end; ++j) {
            for (int k=zz_idx_start; k<zz_idx_end; ++k) {
                h_T_curr[i*y_n*z_n + j*z_n + k] = T;
            }
        }
    }

    // Open file 
    std::ofstream file;
    if (save_data) {
        std::string f_name = "cuda_avg_temp-";
        f_name += "points" + std::to_string(x_n*y_n*z_n)+"-";
        f_name += "n_t" + std::to_string(t_n)+".txt";

        file.open(f_name);
        if (!file.is_open()) {
            std::cerr << "Error! " << f_name << " cannot be open!" << std::endl;
            return 1;
        } else {
            file << "iterat. internal-T external-T elapsed-time" << std::endl;
        }
    }

    // Allocate space on device
    float *d_T_curr;
    float *d_T_next;
    float2 *d_R;
    int *d_heater_id;
    cudaMalloc((void**)&d_T_curr, dim*sizeof(float));
    cudaMalloc((void**)&d_T_next, dim*sizeof(float));
    cudaMalloc((void**)&d_R, n_blocks*sizeof(float2));
    cudaMalloc((void**)&d_heater_id, 6*sizeof(int));

    // Copy data on GPU
    cudaMemcpy(d_T_curr, h_T_curr.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_next, h_T_next.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heater_id, h_heater_id, 6*sizeof(int), cudaMemcpyHostToDevice);

    // Simulation loop
    for (int tau=0; tau<=t_n; ++tau) {
        // Execute simulation loop
        if (use_tiling)
            optimal_loop_simulation<<<simulGridSize, blockSize>>>(d_T_curr, d_T_next, d_heater_id, grid);
        else
            naive_simulation_loop<<<simulGridSize, blockSize>>>(d_T_curr, d_T_next, d_heater_id, grid);

        if (tau%200==0 && show_verbosity) {
            // Compute avg temperatures with kernels
            compute_temperatures<<<tempGridSize, blockSize>>>(d_T_curr, d_R, grid, d_heater_id);
            cudaMemcpy(h_R.data(), d_R, n_blocks*sizeof(float2), cudaMemcpyDeviceToHost);
            float t_in = 0, t_out = 0;
            for (int i=0; i < n_blocks; i++) {
                t_in += h_R[i].y/p_in;
                t_out += h_R[i].x/(dim-p_in);
            }
            printf("\nIteration: %d ----- Tin = %1.4f ----- Tout = %1.4f\n", tau, t_in, t_out);
            
            // Save avg temperature and time
            if (save_data) {
                //file <<  tau << " " << t_in << " " << t_out  << " " << omp_get_wtime()-start << "\n";
            }

            // // Compute avg temperatures in a non-optimal way (just to debug)
            // cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(float), cudaMemcpyDeviceToHost);
            // float avg_t = 0;

            // for (int i=xx_idx_start; i<xx_idx_end; ++i) {
            //     for (int j=yy_idx_start; j<yy_idx_end; ++j) {
            //         for (int k=zz_idx_start; k<zz_idx_end; ++k) {
            //             avg_t += h_T_curr[i*y_n*z_n + j*z_n + k];
            //         }
            //     }
            // }
            // avg_t /= p_in;
            // printf("(SEQUENTIAL/INTERNAL) T(%d) = %1.3e\n", tau, avg_t);
        }

        std::swap(d_T_curr, d_T_next);
    }

    cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(float), cudaMemcpyDeviceToHost);

}