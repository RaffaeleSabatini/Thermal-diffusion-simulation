#include <omp.h>
#include <stdio.h>

int main() {
    printf("Max numero threads: %d\n", omp_get_max_threads());    
    return 0;
}