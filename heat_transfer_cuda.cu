#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cuda_runtime.h>

constexpr int THREADS_PER_SIDE = 8;
constexpr int THREADS_PER_BLOCK = THREADS_PER_SIDE * THREADS_PER_SIDE * THREADS_PER_SIDE;
constexpr int PAD = 2;

//
// ------------------------------------------------------------------------------------------------------------------
//

// Encapsulate grid infos in a single structure
struct Grid {
    int x_n, y_n, z_n;
    double dx, dy, dz, dt;
    double a_w, a_m;

    Grid(int x_n, int y_n, int z_n, double dx, double dy, double dz, double dt, double a_w, double a_m) : 
        x_n(x_n), y_n(y_n), z_n(z_n),
        dx(dx), dy(dy), dz(dz), dt(dt),
        a_w(a_w), a_m(a_m) {}
};

// Taking count of cumulative quantities across iterations
struct Counter {
    double sum_in, sum_out;
    int count_in, count_out;
};

//
// ------------------------------------------------------------------------------------------------------------------
//

// Simulation loop without shared memory
__global__ void naive_simulation_loop(const double* T_curr, double* T_next, const int* h_pos, const Grid grid) {
    int x_id = blockIdx.x * blockDim.x + threadIdx.x;
    int y_id = blockIdx.y * blockDim.y + threadIdx.y;
    int z_id = blockIdx.z * blockDim.z + threadIdx.z;

    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const double dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const double a_w = grid.a_w, a_m = grid.a_m;
    
    if ((y_id > 0 && y_id < y_n-1) && (x_id > 0 && x_id < x_n-1) && (z_id > 0 && z_id < z_n-1)) {
        int t_idx = x_id*y_n*z_n + y_id*z_n + z_id;
        double laplacian = (
            (T_curr[t_idx + y_n*z_n] - 2*T_curr[t_idx] + T_curr[t_idx - y_n*z_n])/dx2 + 
            (T_curr[t_idx + z_n]     - 2*T_curr[t_idx] + T_curr[t_idx - z_n])/(dy2) + 
            (T_curr[t_idx + 1]       - 2*T_curr[t_idx] + T_curr[t_idx - 1])/(dz2) 
        );

        // Use correct thermal diffusivity depending on position
        double a;
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
__global__ void optimal_loop_simulation(const double* T_curr, double* T_next, const int* h_pos, const Grid grid) {
    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const double dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const double a_w = grid.a_w, a_m = grid.a_m;

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
    __shared__ double stencil_T[THREADS_PER_SIDE+PAD][THREADS_PER_SIDE+PAD][THREADS_PER_SIDE+PAD];

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
        double laplacian = (
            (stencil_T[sid_x+1][sid_y][sid_z] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x-1][sid_y][sid_z])/dx2 + 
            (stencil_T[sid_x][sid_y+1][sid_z] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x][sid_y-1][sid_z])/dy2 + 
            (stencil_T[sid_x][sid_y][sid_z+1] - 2*stencil_T[sid_x][sid_y][sid_z] + stencil_T[sid_x][sid_y][sid_z-1])/dz2 
        );

        // Use correct thermal diffusivity depending on position
        double a;
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

__global__ void compute_temperatures(const double* T, Counter* R, const Grid grid, const int* h_pos) {
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
    __shared__ Counter partial_sum[THREADS_PER_BLOCK];

    // In c we'll store 8-reduced values of cumulative T and access counts
    Counter c;
    c.sum_in = 0;
    c.sum_out = 0;
    c.count_in = 0;
    c.count_out = 0;

    // Compilator unrolls nested loops into sequential assembly code
    #pragma unroll
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            for (int k=0; k<2; ++k) {
                
                int t_tid_x = start_x + tid_x + (i*blockDim.x);
                int t_tid_y = start_y + tid_y + (j*blockDim.y);
                int t_tid_z = start_z + tid_z + (k*blockDim.z);

                bool interior_thr = 
                    (t_tid_x > 0 && t_tid_x < x_n-1) && 
                    (t_tid_y > 0 && t_tid_y < y_n-1) && 
                    (t_tid_z > 0 && t_tid_z < z_n-1);

                if (interior_thr) {
                    double val = T[t_tid_x*y_n*z_n + t_tid_y*z_n + t_tid_z];

                    bool inside = (t_tid_x >= h_xmin && t_tid_x < h_xmax) &&
                                  (t_tid_y >= h_ymin && t_tid_y < h_ymax) && 
                                  (t_tid_z >= h_zmin && t_tid_z < h_zmax);
                    
                    if (inside) {
                        c.sum_in += val;
                        c.count_in += 1;
                    }
                    else {
                        c.sum_out += val;
                        c.count_out += 1;
                    } 
                }
            }
        }
    }

    // Load shared memory with partial sums
    partial_sum[s_tid] = c;
    __syncthreads();

    // Performing reduction
    for (int stride = 2; stride <= THREADS_PER_BLOCK; stride *= 2) {
        if (s_tid < THREADS_PER_BLOCK/stride) {
            partial_sum[s_tid].sum_in  += partial_sum[s_tid + THREADS_PER_BLOCK/stride].sum_in;
            partial_sum[s_tid].sum_out += partial_sum[s_tid + THREADS_PER_BLOCK/stride].sum_out;

            partial_sum[s_tid].count_in  += partial_sum[s_tid + THREADS_PER_BLOCK/stride].count_in;
            partial_sum[s_tid].count_out += partial_sum[s_tid + THREADS_PER_BLOCK/stride].count_out;
        }
        __syncthreads();
    }

    if (s_tid == 0) {
        R[blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z] = partial_sum[0];
    }
}

//
// ------------------------------------------------------------------------------------------------------------------
//

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
    const double x_len = 10., y_len = 10., z_len = 10.;
    const double xx_len = 2, yy_len = 2, zz_len = 2;
    const double a_w = 0.143, a_c = 111;

    // Position of internal heater
    const double xx_pos = (x_len-xx_len)/2;
    const double yy_pos = (y_len-yy_len)/2;
    const double zz_pos = (z_len-zz_len)/2;

    // T: heater temperature     T0: water temperature
    double T = 300, T0 = 280;
    double dx = x_len/x_n, dy = y_len/y_n, dz = z_len/z_n, dt = 1e-6;  
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

    printf("• Points along (x,y,z) in the internal system: %d, %d, %d (total: %d)\n\n", xx_idx_end-xx_idx_start, yy_idx_end-yy_idx_start, zz_idx_end-zz_idx_start, p_in);
    
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

    std::vector<double> h_T_curr(dim, T0);
    std::vector<double> h_T_next(dim, T0);
    std::vector<Counter> h_R(n_blocks);

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
        std::string f_name = "cuda_avg_temp";
        
        if (use_tiling) f_name += "-tiling";
        
        f_name += 
        "-x" + std::to_string(x_n) + 
        "-y" + std::to_string(y_n) +
        "-z" + std::to_string(z_n) +
        "-t" + std::to_string(t_n);

        f_name += ".txt";

        file.open(f_name);
        if (!file.is_open()) {
            std::cerr << "Error! " << f_name << " cannot be open!" << std::endl;
            return 1;
        } else {
            file << "iteration internal-T external-T loop-time(ms) wall-time(ms)" << std::endl;
        }
    }

    // Allocate space on device
    double *d_T_curr;
    double *d_T_next;
    Counter *d_R;
    int *d_heater_id;
    cudaMalloc((void**)&d_T_curr, dim*sizeof(double));
    cudaMalloc((void**)&d_T_next, dim*sizeof(double));
    cudaMalloc((void**)&d_R, n_blocks*sizeof(Counter));
    cudaMalloc((void**)&d_heater_id, 6*sizeof(int));

    // Copy data on GPU
    cudaMemcpy(d_T_curr, h_T_curr.data(), dim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_next, h_T_next.data(), dim*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heater_id, h_heater_id, 6*sizeof(int), cudaMemcpyHostToDevice);

    // Measure wall time
    auto simulation_s = std::chrono::system_clock::now();
    
    // Simulation loop
    for (int tau=0; tau<=t_n; ++tau) {
        // Measure time for benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float loop_time, calcT_time;

        // Execute simulation loop
        cudaEventRecord(start, 0);
        if (use_tiling)
            optimal_loop_simulation<<<simulGridSize, blockSize>>>(d_T_curr, d_T_next, d_heater_id, grid);
        else
            naive_simulation_loop<<<simulGridSize, blockSize>>>(d_T_curr, d_T_next, d_heater_id, grid);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Compute elapsed time
        cudaEventElapsedTime(&loop_time, start, stop);
        
        // ----------------------------------------------------------------------
        // AVERAGE TEMPERATURES
        // ----------------------------------------------------------------------
        cudaEventRecord(start, 0);
        compute_temperatures<<<tempGridSize, blockSize>>>(d_T_curr, d_R, grid, d_heater_id);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Compute elapsed time
        cudaEventElapsedTime(&calcT_time, start, stop);

        // Computing averages
        cudaMemcpy(h_R.data(), d_R, n_blocks*sizeof(Counter), cudaMemcpyDeviceToHost);

        double t_in = 0, t_out = 0;
        int count_in = 0, count_out = 0;

        for (int i=0; i < n_blocks; i++) {
            t_in  += h_R[i].sum_in;
            t_out += h_R[i].sum_out;
            count_in += h_R[i].count_in;
            count_out += h_R[i].count_out;
        }
        t_in  /= count_in;
        t_out /= count_out;

        if (show_verbosity) {
            printf("\nIteration: %d\n", tau);
            printf("T_in = %1.4f ----- T_out-T0 = %1.3e\n", t_in, t_out-T0);
            printf("Loop time = %1.4f ----- Computing-T time = %1.4f\n", loop_time, calcT_time);
            printf("----------------------------------------------------\n");

            // // Compute avg temperatures in a non-optimal way (just to debug)
            // cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(double), cudaMemcpyDeviceToHost);
            // double avg_t = 0;

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
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Save avg temperature and time
        if (save_data) {
            auto simulation_e = std::chrono::system_clock::now();
            auto wallt = std::chrono::duration_cast<std::chrono::milliseconds>(simulation_e-simulation_s); 
            file << std::setprecision(std::numeric_limits<double>::max_digits10) <<
                tau << " " << 
                t_in << " " << 
                t_out  << " " << 
                loop_time << " " << 
                wallt.count() << 
                "\n";
        }

        // Swap tensors to use d_T_next as a buffer
        std::swap(d_T_curr, d_T_next);
    }

    cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(double), cudaMemcpyDeviceToHost);

}