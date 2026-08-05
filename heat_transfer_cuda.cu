#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>

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

__global__ void naive_simulation_loop(const float* T_curr, const float* T_next, const int* h_pos, const Grid& grid) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;

    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const float dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const float a_w = grid.a_w, a_m = grid.a_m;
    
    if ((row > 0 && row < y_n-1) && (col > 0 && col < x_n-1) && (dep > 0 && dep < z_n-1)) {
        int idx = dep*y_n*x_n + col*x_n + row;
        float laplacian = (
            (T_curr[idx + y_n*z_n] - 2*T_curr[idx] + T_curr[idx - y_n*z_n])/dx2 + 
            (T_curr[idx + z_n]     - 2*T_curr[idx] + T_curr[idx - z_n])/(dy2) + 
            (T_curr[idx + 1]       - 2*T_curr[idx] + T_curr[idx - 1])/(dz2) 
        );

        // Use correct thermal diffusivity depending on position
        float a;
        if ((i>=h_pos[0] && i<h_pos[1]) && 
            (j>=h_pos[2] && j<h_pos[3]) &&
            (k>=h_pos[4] && k<h_pos[5])) 
            a = a_m; 
        else
            a = a_w;

        // Update rule
        T_next[idx] = T_curr[idx] + a*dt*laplacian;

        // Manca da aggiornare avg_T
        // ...
    }
}

int main(int* argc, char* argv[]) {
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

    // We add a heater at temperature T0
    float T0 = 300;
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
    std::vector<float> h_T_curr(dim, 280);
    std::vector<float> h_T_next(dim, 280);

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
    cudaMalloc((void**)&d_T_curr, dim*sizeof(float));
    cudaMalloc((void**)&d_T_next, dim*sizeof(float));

    // Copy data on GPU
    cudaMemcpy(d_T_curr, h_T_curr.data(), dim*sizeof(float), cudaMemcpyHostToDevice);

    // Simulation loop
    dim3 blockSize(16, 16, 16);
    dim3 gridSize(
        (x_n + blockSize.x - 1)/blockSize.x,
        (y_n + blockSize.y - 1)/blockSize.y,
        (z_n + blockSize.z - 1)/blockSize.z
    )
    for (int tau=0; tau<t_n; ++tau) {
        // Esegui simulation loop
        // ...
        // Swap delle temperature
        // ...
    }

}
