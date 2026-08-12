#include <stdio.h>
#include <vector>
#include <string>
#include <utility>
#include <math.h>
#include <cuda_runtime.h>

constexpr int THREADS_PER_SIDE = 8;
constexpr int THREADS_PER_BLOCK = (int) std::pow(THREADS_PER_SIDE, 3);

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

//__global__ void optimal_loop_simulation


__global__ void compute_temperatures(const float* T, float* R, const Grid grid, const* int h_pos) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;

    int start_x = 2 * blockDim.x * blockIdx.x;
    int start_y = 2 * blockDim.y * blockIdx.y;
    int start_z = 2 * blockDim.z * blockIdx.z;

    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;

    int start_id = start_x*y_n*z_n + start_y*z_n + start_z;
    int tid      = tid_x * blockDim.y * blockDim.z + tid_y * blockDim.z + tid_z;

    // Fill shared memory 
    __shared__ float partial_sum[8*THREADS_PER_BLOCK];

    // Threads block (i,j,k)-th reads tensor blocks (i+a,j+a,k+a)-th where a=0,1
    partial_sum[tid] = T[start_id + tid]; // (i,j,k)
    partial_sum[tid + blockDim.z] = T[start_id + tid + blockDim.z]; // (i,j,k+1)
    partial_sum[tid + blockDim.y * blockDim.z] = T[start_id + tid + blockDim.y * blockDim.z]; // (i,j+1,k)
    partial_sum[tid + blockDim.x * blockDim.y * blockDim.z] = T[start_id + tid + blockDim.x * blockDim.y * blockDim.z]; // (i+1,j,k)
    partial_sum[tid + (blockDim.y+1) * blockDim.z] = T[start_id + tid + (blockDim.y+1) * blockDim.z]; // (i,j+1,k+1)
    partial_sum[tid + (blockDim.y*blockDim.x+1) * blockDim.z] = T[start_id + tid + (blockDim.y*blockDim.x+1) * blockDim.z]; // (i+1,j,k+1)
    partial_sum[tid + (blockDim.x+1) * blockDim.y * blockDim.z] = T[start_id + tid + (blockDim.x+1) * blockDim.y * blockDim.z]; // (i+1,j+1,k)
    partial_sum[tid + blockDim.z * (1 + blockDim.y * (1 + blockDim.x))] = T[start_id + tid + blockDim.z * (1 + blockDim.y * (1 + blockDim.x))]; // (i+1,j+1,k+1)

    // Performing reduction along each axis separately
    // Along z-axis
    for (int stride=1; stride <= blockDim.z; stride*=2) {
        __syncthreads();
        if (tid_z < blockDim.z/stride) {
            partial_sum[tid] += partial_sum[tid + blockDim.z/stride];
            partial_sum[tid + blockDim.y * blockDim.z] += partial_sum[tid + blockDim.y * blockDim.z + blockDim.z/stride];
            partial_sum[tid + blockDim.x * blockDim.y * blockDim.z] += partial_sum[tid + blockDim.x * blockDim.y * blockDim.z + blockDim.z/stride];
            partial_sum[tid + (1+blockDim.x) * blockDim.y * blockDim.z] += partial_sum[tid + (1+blockDim.x) * blockDim.y * blockDim.z + blockDim.z/stride];
        }
    }

    // Along y-axis
    for (int stride=1; stride <= blockDim.y; stride*=2) {
        __syncthreads();
        if ((tid_y < blockDim.y/stride) && tid_z==0) { // Possibile errore: tid_z = 0 o 1?
            partial_sum[tid] += partial_sum[tid + blockDim.y*blockDim.z/stride];
            partial_sum[tid + blockDim.x * blockDim.y * blockDim.z] += partial_sum[tid + blockDim.x * blockDim.y * blockDim.z + blockDim.y*blockDim.z/stride];
        }
    }

    // Along x-axis
    for (int stride=1; stride<= blockDim.x; stride*=2) {
        __syncthreads();
        if ((tid_x < blockDim.x/stride) && tid_z==0 && tid_y==0) {
            partial_sum[tid] += partial_sum[tid + blockDim.x * blockDim.y * blockDim.z/stride];
        }
    }

    __syncthreads();
    if (tid == 0) {
        R[blockIdx.x * blockDim.y * blockDim.z + blockIdx.y * blockDim.z + blockIdx.z] = partial_sum[0];
    }
}

int main(int argc, char* argv[]) {
    // Grid specifications
    int t_n = 2000;
    int x_n = 128, y_n = 128, z_n = 128;
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

    int dim = x_n*y_n*z_n;
    int n_blocks = (dim + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    std::vector<float> h_T_curr(dim, T);
    std::vector<float> h_T_next(dim, T);
    std::vector<float> h_R(n_blocks, 0);
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
                h_T_curr[i*y_n*z_n + j*z_n + k] = T0;
            }
        }
    }

    // Allocate space on device
    float *d_T_curr;
    float *d_T_next;
    float *d_R;
    int *d_heater_id;
    cudaMalloc((void**)&d_T_curr, dim*sizeof(float));
    cudaMalloc((void**)&d_T_next, dim*sizeof(float));
    cudaMalloc((void**)&d_R, n_blocks*sizeof(float));
    cudaMalloc((void**)&d_heater_id, 6*sizeof(int));

    // Copy data on GPU
    cudaMemcpy(d_T_curr, h_T_curr.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T_next, h_T_next.data(), dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heater_id, h_heater_id, 6*sizeof(int), cudaMemcpyHostToDevice);

    // Simulation loop
    dim3 blockSize(THREADS_PER_SIDE, THREADS_PER_SIDE, THREADS_PER_SIDE);
    dim3 gridSize(
        (x_n + blockSize.x - 1)/blockSize.x,
        (y_n + blockSize.y - 1)/blockSize.y,
        (z_n + blockSize.z - 1)/blockSize.z
    );
    for (int tau=0; tau<t_n; ++tau) {
        // Execute simulation loop
        naive_simulation_loop<<<gridSize, blockSize>>>(d_T_curr, d_T_next, d_heater_id, grid);

        if (tau%200==0) {
            // Compute avg temperatures with kernels
            compute_temperatures(d_T_curr, d_R, grid, h_pos);
            cudaMemcpy(h_R.data(), d_R, n_blocks*sizeof(float), cudaMemcpyDeviceToHost);
            float avg_t = 0;
            for (auto& r: h_R) {
                avg_t += r;
            }
            avg_t /= dim;
            printf("(KERNEL/TOTAL) T(%d) = %1.3e\n", tau, avg_t);

            // Compute avg temperatures in a non-optimal way (just to debug)
            cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(float), cudaMemcpyDeviceToHost);
            float avg_t = 0;

            for (int i=xx_idx_start; i<xx_idx_end; ++i) {
                for (int j=yy_idx_start; j<yy_idx_end; ++j) {
                    for (int k=zz_idx_start; k<zz_idx_end; ++k) {
                        avg_t += h_T_curr[i*y_n*z_n + j*z_n + k];
                    }
                }
            }
            avg_t /= p_in;
            printf("(SEQUENTIAL/INTERNAL) T(%d) = %1.3e\n", tau, avg_t);
        }

        std::swap(d_T_curr, d_T_next);
    }

    cudaMemcpy(h_T_curr.data(), d_T_curr, dim*sizeof(float), cudaMemcpyDeviceToHost);

}
