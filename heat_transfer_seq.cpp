#include <stdio.h>
#include <math.h>
#include <utility>
#include <vector>

int main(int argc, char *argv[]) {

    // Grid specifications
    int t_n = 1000;
    int x_n = 100, y_n = 100, z_n = 100;
    int tol = 10;

    // Systems physical dimensions (space: mm, time: sec)
    const float x_len = 10., y_len = 10., z_len = 10., t_len = 0.001;
    const float xx_len = 2, yy_len = 2, zz_len = 2;
    const float a_w = 0.143, a_c = 111;

    // Position of internal heater
    const float xx_pos = (x_len-xx_len)/2;
    const float yy_pos = (y_len-yy_len)/2;
    const float zz_pos = (z_len-zz_len)/2;

    // Temperature field 
    std::vector<double> T_curr(x_n*y_n*z_n, 273);
    std::vector<double> T_next(x_n*y_n*z_n, 273);

    // We add a heater at temperature T0
    float T0 = 300;
    float dx = x_len/x_n, dy = y_len/y_n, dz = z_len/z_n, dt = t_len/t_n;
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
                T_curr[i*y_n*z_n + j*z_n + k] = T0;
            }
        }
    }

    // Simulation loop
    for (int tau=0; tau<t_n; tau++) {
        float sum = 0;

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
                        sum += T_curr[i*y_n*z_n+j*z_n+k];
                    } else {
                        a = a_w;
                    }

                    // Update rule
                    T_next[i*y_n*z_n+j*z_n+k] = T_curr[i*y_n*z_n+j*z_n+k] + a*dt*laplacian;
                }  
            }
        }
        std::swap(T_curr, T_next);

        // Compute cube avg temperature
        if (tau % 10 == 0) {
            double avg_t = sum/p_in;
            printf("Avg temperature of the cube at iteration/time (%d/%1.3e): (%1.3f)\n", tau, tau*dt, avg_t);
        }
    }

    return 0;

}