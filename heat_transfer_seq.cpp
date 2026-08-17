#include <stdio.h>
#include <math.h>
#include <utility>
#include <vector>
#include <iomanip>

int main(int argc, char *argv[]) {

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
    std::vector<double> T_curr(x_n*y_n*z_n, T0);
    std::vector<double> T_next(x_n*y_n*z_n, T0);

    double dx = x_len/x_n, dy = y_len/y_n, dz = z_len/z_n, dt = 1e-6;
    printf("\n---------------------------------------\n\n");
    printf("Running heat transfer simulation with:\n\n");
    printf("Points along (x,y,z): %d, %d, %d (total: %d)\n\n", x_n, y_n, z_n, x_n*y_n*z_n);
    printf("Step-size along (x,y,z,t): %1.3f, %1.3f, %1.3f, %1.3e\n\n", dx, dy, dz, dt);

    
    if (xx_len/dx<tol) {
        printf("Error: not enough points (%d) along x in the small system.", (int)(xx_len/dx));
    } else if (yy_len/dy<tol) {
        printf("Error: not enough points (%d) along y in the small system.", (int)(yy_len/dy));
    } else if (zz_len/dz<tol) {
        printf("Error: not enough points (%d) along z in the small system.", (int)(z_len/dz));
    }

    // Filling temperature field with temperature of heater
    int xx_idx_start = xx_pos/dx, xx_idx_end = (xx_pos+xx_len)/dx;
    int yy_idx_start = yy_pos/dy, yy_idx_end = (yy_pos+yy_len)/dy;
    int zz_idx_start = zz_pos/dz, zz_idx_end = (zz_pos+zz_len)/dz;
    int p_in = (xx_idx_end-xx_idx_start)*(yy_idx_end-yy_idx_start)*(zz_idx_end-zz_idx_start);

    printf("Points along (x,y,z) in the internal system: %d, %d, %d (total: %d)\n\n", xx_idx_end-xx_idx_start, yy_idx_end-yy_idx_start, zz_idx_end-zz_idx_start, p_in);

    for (int i=xx_idx_start; i<xx_idx_end; ++i) {
        for (int j=yy_idx_start; j<yy_idx_end; ++j) {
            for (int k=zz_idx_start; k<zz_idx_end; ++k) {
                T_curr[i*y_n*z_n + j*z_n + k] = T;
            }
        }
    }

    // Simulation loop
    for (int tau=0; tau<=t_n; tau++) {
        // Note: superficial points are not iterated over in the loop because are kept fixed by boundary conditions
        double sum_out = 0, sum_in  = 0;
        int count_in = 0, count_out = 0;
          
        // Boundary conditions: surface of the box is at constant T ----> Not updated
        for (int i=1; i<(x_n-1); ++i) {
            for (int j=1; j<(y_n-1); ++j) {
                for (int k=1; k<(z_n-1); ++k) {

                    double laplacian = (
                        (T_curr[(i+1)*y_n*z_n+j*z_n+k] - 2*T_curr[i*y_n*z_n+j*z_n+k] + T_curr[(i-1)*y_n*z_n+j*z_n+k])/(dx*dx) + 
                        (T_curr[i*y_n*z_n+(j+1)*z_n+k] - 2*T_curr[i*y_n*z_n+j*z_n+k] + T_curr[i*y_n*z_n+(j-1)*z_n+k])/(dy*dy) + 
                        (T_curr[i*y_n*z_n+j*z_n+(k+1)] - 2*T_curr[i*y_n*z_n+j*z_n+k] + T_curr[i*y_n*z_n+j*z_n+(k-1)])/(dz*dz) 
                    );

                    // Use correct thermal diffusivity depending on position
                    double a;
                    if (
                        (i>=xx_idx_start && i<xx_idx_end) && 
                        (j>=yy_idx_start && j<yy_idx_end) &&
                        (k>=zz_idx_start && k<zz_idx_end)
                    ) {    
                        a = a_c; 

                        // We measure the average temperature of the heater
                        sum_in += T_curr[i*y_n*z_n+j*z_n+k];
                        count_in += 1;
                    } else {
                        a = a_w;

                        // Temperature of the water
                        sum_out += T_curr[i*y_n*z_n+j*z_n+k];
                        count_out += 1;
                    }

                    // Update rule
                    T_next[i*y_n*z_n+j*z_n+k] = T_curr[i*y_n*z_n+j*z_n+k] + a*dt*laplacian;
                }  
            }
        }
        std::swap(T_curr, T_next);

        // Compute cube avg temperature
        double T_in  = sum_in/count_in;
        double T_out = sum_out/count_out;
        printf("Iter. %d ------ T_in (K): %1.6f ------ T_out-T0 (K): %1.3e\n", tau, T_in, T_out-T0);
    }

    return 0;
}