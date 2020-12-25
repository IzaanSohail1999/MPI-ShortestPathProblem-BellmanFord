#include <string>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include "mpi.h"
using namespace std;
using std::string;
using std::cout;
using std::endl;
#define INFINITY 1000000
int* Distance;
#define N 5 //number of vertices
int Dimension_converter_from2Dto1D(int x, int y, int n) {
    return x * n + y;
}
int print_result(bool neg_cycle, int* dist)
{

    if (!neg_cycle) {
        for (int i = 0; i < N; i++) {
            if (dist[i] > INFINITY)
                dist[i] = INFINITY;
            cout << dist[i] << "\n";
        }
    }
    else {
        cout<<"FOUND NEGATIVE CYCLE!" << endl;
    }
    return 0;
}
void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int matrix[][N], int* dist, bool* has_negative_cycle) {
    int copy_of_var_N; // need a local copy for N
    int loc_start, loc_end;
    int* matrixA; //local matrix
    int* my_Dist; //local distance

    // broadcast N
    if (my_rank == 0) {
        copy_of_var_N = n;
    }
    MPI_Bcast(&copy_of_var_N, 1, MPI_INT, 0, comm);

    // find local task range
    int m = copy_of_var_N / p;
    loc_start = m * my_rank;
    loc_end = m * (my_rank + 1);
    if (my_rank == p - 1) {
        loc_end = copy_of_var_N;
    }

    //allocate local memory
    matrixA = (int*)malloc(copy_of_var_N * copy_of_var_N * sizeof(int));
    my_Dist = (int*)malloc(copy_of_var_N * sizeof(int));

    //step 4: broadcast matrix mat
    if (my_rank == 0)
        memcpy(matrixA, matrix, sizeof(int) * copy_of_var_N * copy_of_var_N);
    MPI_Bcast(matrixA, copy_of_var_N * copy_of_var_N, MPI_INT, 0, comm);

    //bellman-ford algorithm
    for (int i = 0; i < copy_of_var_N; i++) {
        my_Dist[i] = INFINITY;
    }
    my_Dist[0] = 0;
    MPI_Barrier(comm);

    bool location_change;
    int i = 0;
    for (int iter = 0; iter < copy_of_var_N - 1; iter++) {
        location_change = false;
        i++;
        for (int u = loc_start; u < loc_end; u++) {
            for (int v = 0; v < copy_of_var_N; v++) {
                int weight = matrixA[Dimension_converter_from2Dto1D(u, v, copy_of_var_N)];
                if (weight < INFINITY) {
                    if (my_Dist[u] + weight < my_Dist[v]) {
                        my_Dist[v] = my_Dist[u] + weight;
                        location_change = true;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &location_change, 1, MPI_INT, MPI_LOR, comm);
        if (!location_change)
            break;
        MPI_Allreduce(MPI_IN_PLACE, my_Dist, copy_of_var_N, MPI_INT, MPI_MIN, comm);
    }

    
    if (i == copy_of_var_N - 1) {
        location_change = false;
        for (int u = loc_start; u < loc_end; u++) {
            for (int v = 0; v < copy_of_var_N; v++) {
                int weight = matrixA[Dimension_converter_from2Dto1D(u, v, copy_of_var_N)];
                if (weight < INFINITY) {
                    if (my_Dist[u] + weight < my_Dist[v]) {
                        my_Dist[v] = my_Dist[u] + weight;
                        location_change = true;
                        break;
                    }
                }
            }
        }
        MPI_Allreduce(&location_change, has_negative_cycle, 1, MPI_INT, MPI_LOR, comm);
    }
    if (my_rank == 0)
        memcpy(dist, my_Dist, copy_of_var_N * sizeof(int));

    free(matrixA);
    free(my_Dist);

}

int main(int argc, char** argv) {
    double time1, time2;

    int matrix[N][N] = {0, -1 , 4 , 1000000 ,1000000 , 1000000 , 0 , 3 ,2,2,1000000 ,1000000, 0 , 1000000, 1000000 , 1000000,1 ,5,0,1000000 ,1000000,1000000,1000000,-3 , 0 };
    bool neg_cycle = false;

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p;//number of processors
    int my_rank;//my global rank
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);


    if (my_rank == 0) {
        Distance = (int*)malloc(sizeof(int) * N);
    }
    MPI_Barrier(comm);
    time1 = MPI_Wtime();
    bellman_ford(my_rank, p, comm, N, matrix, Distance, &neg_cycle);
    MPI_Barrier(comm);
    time2 = MPI_Wtime();

    if (my_rank == 0) {
        cout<<setprecision(6) << "Time(s): " << (time2 - time1) << endl;
        print_result(neg_cycle, Distance);
        free(Distance);
    }
    MPI_Finalize();
    return 0;
}
