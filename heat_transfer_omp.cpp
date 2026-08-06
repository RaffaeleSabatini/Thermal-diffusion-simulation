#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <utility>
#include <vector>
#include <algorithm>
#include <fstream>
#include <omp.h>

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

// Simulation loop at given time without tiling strategy
void naive_simulation_loop(const Grid& grid, std::vector<float>& T_curr, std::vector<float>& T_next, int h_pos[6], float& t_in, float& t_out) {
    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const float dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const float a_w = grid.a_w, a_m = grid.a_m;

    // Boundary conditions: surface of the box is at constant T 
    // => superficial tensor elements are not update
    #pragma omp parallel for collapse(2) reduction(+:avg_t) schedule(static)
    for (int i=1; i<(x_n-1); ++i) {
        for (int j=1; j<(y_n-1); ++j) {
            int ij_idx = i*y_n*z_n + j*z_n;
            for (int k=1; k<(z_n-1); ++k) {
                int idx = ij_idx + k;
                float laplacian = (
                    (T_curr[idx + y_n*z_n] - 2*T_curr[idx] + T_curr[idx - y_n*z_n])/dx2 + 
                    (T_curr[idx + z_n]     - 2*T_curr[idx] + T_curr[idx - z_n])/(dy2) + 
                    (T_curr[idx + 1]       - 2*T_curr[idx] + T_curr[idx - 1])/(dz2) 
                );

                // Use correct thermal diffusivity depending on position
                float a;
                if ((i>=h_pos[0] && i<h_pos[1]) && 
                    (j>=h_pos[2] && j<h_pos[3]) &&
                    (k>=h_pos[4] && k<h_pos[5])) {    
                    a = a_m; 
                    t_in += T_curr[idx];
                } else {
                    a = a_w;
                    t_out += T_curr[idx];
                }

                // Update rule
                T_next[idx] = T_curr[idx] + a*dt*laplacian;
            }  
        }
    }
}   

// Simulation loop with tiling strategy
void tiling_simulation_loop(const Grid& grid, std::vector<float>& T_curr, std::vector<float>& T_next, int h_pos[6], float& t_in, float& t_out) {
    const int x_n=grid.x_n, y_n=grid.y_n, z_n=grid.z_n;
    const float dx2 = grid.dx*grid.dx, dy2 = grid.dy*grid.dy, dz2 = grid.dz*grid.dz, dt = grid.dt;
    const float a_w = grid.a_w, a_m = grid.a_m;
    const int Bx = 16, By = 16, Bz = 16; // Dimensions of tiles

    /* Each block is assigned to one thread:
        - 3 external loops are used to select blocks
        - 3 internal loops are used to select entries 
    */
    #pragma omp parallel for collapse(2) reduction(+:avg_t)
    for (int ii=1; ii<x_n-1; ii+=Bx) {
        for (int jj=1; jj<y_n-1; jj+=By) {
            int i_max = std::min(ii+Bx, x_n-1); // Avoid access out-of-index
            int j_max = std::min(jj+By, y_n-1); // Avoid access out-of-index
            for (int kk=1; kk<z_n-1; kk+=Bz) {
                int k_max = std::min(kk+Bz, z_n-1); // Avoid access out-of-index

                for (int i=ii; i<i_max; ++i) {
                    for (int j=jj; j<j_max; ++j) {
                        int ij_idx = i*y_n*z_n + j*z_n; 
                        for (int k=kk; k<k_max; ++k) {

                            int idx = ij_idx + k;
                            float laplacian = (
                                (T_curr[idx + y_n*z_n] - 2*T_curr[idx] + T_curr[idx - y_n*z_n])/dx2 + 
                                (T_curr[idx + z_n]     - 2*T_curr[idx] + T_curr[idx - z_n])/(dy2) + 
                                (T_curr[idx + 1]       - 2*T_curr[idx] + T_curr[idx - 1])/(dz2) 
                            );

                            // Use correct thermal diffusivity depending on position
                            float a;
                            if ((i>=h_pos[0] && i<h_pos[1]) && 
                                (j>=h_pos[2] && j<h_pos[3]) &&
                                (k>=h_pos[4] && k<h_pos[5])) {    
                                a = a_m; 
                                t_in += T_curr[idx];
                            } else {
                                a = a_w;
                                t_out += T_curr[idx]
                            }

                            // Update rule
                            T_next[idx] = T_curr[idx] + a*dt*laplacian;
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {

    // -------------- List of flags that can be turned on from terminal --------------

    bool save_data = false;        // save in a file temperature in function of time
    bool show_verbosity = false;   // show execution on terminal
    bool use_tiling = false;       // use reduction directive (to implement different ones)
    bool scale_threads = false;    // execute the program for different values of n_threads

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
            } else if (arg == "--scale-thread") {
                scale_threads = true;
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
    
    Grid grid(x_n, y_n, z_n, dx, dy, dz, dt, a_w, a_c);
    int heater_id[6] = {
        xx_idx_start, xx_idx_end,
        yy_idx_start, yy_idx_end,
        zz_idx_start, zz_idx_end
    };

    // If scale_thread == False, we execute the simulation once with max number of threads
    int initial_thr_num = scale_threads ? 1 : omp_get_max_threads();
    int final_thr_num   = omp_get_max_threads();

    for (int n_threads=initial_thr_num; n_threads<=final_thr_num; ++n_threads) {

        // Open file 
        std::ofstream file;
        if (save_data) {
            std::string f_name = "avg_temp-";
            f_name += "n_threads" + std::to_string(n_threads)+"-";
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

        printf("Running simulation with thread num. (%d/%d)\n", n_threads, final_thr_num);
        omp_set_num_threads(n_threads);

        // Temperature field 
        std::vector<float> T_curr(x_n*y_n*z_n, 280);
        std::vector<float> T_next(x_n*y_n*z_n, 280);

        // Setting initial temperature of the internal heater
        for (int i=xx_idx_start; i<xx_idx_end; ++i) {
            for (int j=yy_idx_start; j<yy_idx_end; ++j) {
                for (int k=zz_idx_start; k<zz_idx_end; ++k) {
                    T_curr[i*y_n*z_n + j*z_n + k] = T0;
                }
            }
        }

        // Executing simulation 
        double start = omp_get_wtime();
        for (int tau=0; tau<t_n; tau++) {

            // avg temperatures of internal block and external material
            float t_in = 0;
            float t_out = 0;

            if (use_tiling)
                tiling_simulation_loop(grid, T_curr, T_next, heater_id, t_in, t_out);
            else
                naive_simulation_loop(grid, T_curr, T_next, heater_id, t_in, t_out);

            std::swap(T_curr, T_next);
            avg_t /= p_in;

            // Show avg temperature
            if ((tau % 100 == 0) && show_verbosity) {
                printf("Iteration: %d/%d ------ Time: %1.3e ------ Internal Temp.: (%1.3f)\n", tau, t_n, tau*dt, t_in);
            }

            // Save avg temperature and time
            if (save_data) {
                file <<  tau << " " << t_in << " " << t_out << omp_get_wtime()-start << "\n";
            }
        }

        double t_final = omp_get_wtime() - start;
        printf("Simulation completed in %1.3f seconds.\n", t_final);
        printf("\n------------------------------------------------------\n");
    }
    
    return 0;

}