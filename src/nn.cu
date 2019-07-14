#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "gputimer.h"
#include "gpuerrors.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <math.h>

#define D 128
#define D_L 100
#define N_ref 1000000


#define  TILEX  50
#define  TILEY  16
#define  TILEZ  128//it is for 128 dim of input arry


#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y 
#define bz blockIdx.z

#define section 50

#define thread_x1 32
#define thread_y1 32



// ===========================> Functions Prototype <===============================
int fvecs_read (const char *fname, int d, int n, float *a);
int ivecs_write (const char *fname, int d, int n, const int *v);
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K);
void gpuKernels(float* ref, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time);
// =================================================================================

int main(int argc, char *argv[]) {

    struct cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("Device Name: %s\n", p.name);

    // get parameters from command line
    unsigned int N, K;
    get_inputs(argc, argv, N, K);

    // allocate memory in CPU for calculation
    float* reference; // reference vectors
    float* query; // query points
    int* hist;

    // Memory Allocation
	
	
	reference=(float*)malloc(sizeof(float)*D*N_ref);
	query=(float*)malloc(sizeof(float)*D*N);
	hist=(int*)malloc(sizeof(int)*N*K);
	
	


    // fill references, query and labels with the values read from files
    fvecs_read("/home/data/ref.fvecs", D, N_ref, reference);
    fvecs_read("/home/data/query.fvecs", D, N, query);
    
    // time measurement for GPU calculation
    double gpu_kernel_time = 0.0;
    clock_t t0 = clock();
	  gpuKernels(reference, query, hist, N, K, &gpu_kernel_time);
    clock_t t1 = clock();

    printf("k=%d n=%d GPU=%g ms GPU-Kernels=%g ms\n",
    K, N, (t1-t0)/1000.0, gpu_kernel_time);

    // write the output to a file
    ivecs_write("outputs.ivecs", K, N, hist);
    
    // free allocated memory for later use
    free(reference);
    free(hist);
    free(query);

    return 0;
}
//-----------------------------------------------------------------------------
__global__ void distance_kernel(float* gpu_refrence,float* gpu_query, float* gpu_distance){

int row=by*TILEX+ty;
int colm=bx*TILEY+tx;


float sub=0;
float accum=0;
int i=0;

__shared__ float query_shared[TILEZ][TILEX];
__shared__ float refrence_shared[TILEZ][TILEY];


    for(int count=0;count<128/TILEZ;count++){

	for(i=0;i<TILEZ/TILEY;i++)query_shared[i*TILEY+tx][ty]=gpu_query[row*128+count*TILEZ+i*TILEY+tx];
	
	for(i=ty;i<TILEZ;i=TILEX+i)refrence_shared[i][tx]=gpu_refrence[colm*128+count*TILEZ+i];
	
	__syncthreads();
	
	for(i=0;i<TILEZ;i++){
                            sub=(query_shared[i][ty]-refrence_shared[i][tx]);
                            accum+=sub*sub;
                             }
	
        }


gpu_distance[row*N_ref+colm]=sqrt(accum);

}
//-----------------------------------------------------------------------------
__global__ void binarysearch_kernel(float* gpu_distance,int* gpu_temp,int K){
	

    int temp_base=by*K;
	int distance_base=by*N_ref;
	int bind_index=bx*1024+ty*32+tx;
	

	if(bind_index<K){
		
			
			float min=gpu_distance[distance_base];
			float max=gpu_distance[distance_base+N_ref-1];
			float interval=(max-min)/K;
			float bind=min+interval*bind_index;
			int count=(1<<19);
			int offset=48576;
                        

						
		for(int i=0;i<19;i++){
			
			
			if((count-offset-1)<0){
				
				
				count=count+(1<<(18-i));
		
				
				
			}

		else if(bind<gpu_distance[distance_base+count-offset-1]){
				
				

                                count=count-(1<<(19-i))+(1<<(18-i));
				
			}
			
			else{
				
				count=count+(1<<(18-i));

			}

			
		}
		
		
		gpu_temp[temp_base+bind_index]=count-offset;
	
    }
		
	
}


//-----------------------------------------------------------------------------
__global__ void sub_kernel(int* gpu_temp,int* gpu_hist,int K){
	
	
	int base=by*K;
	int index=bx*1024+ty*32+tx;
	
	if(index<K-1){
		
		gpu_hist[base+index]=gpu_temp[base+index+1]-gpu_temp[base+index];
		
	}
	
	if(index==K-1)gpu_hist[base+index]=N_ref+1-gpu_temp[base+index];
		
}


//-----------------------------------------------------------------------------
void gpuKernels(float* reference, float* query, int* hist, unsigned int N, unsigned int K, double* gpu_kernel_time) {

    // Memory Allocation and Copy to Device



	
	float* gpu_refrence;
	float* gpu_query;
	float* gpu_distance;
	int* gpu_temp;
	int* gpu_hist;

		
    HANDLE_ERROR(cudaMalloc((void**)&gpu_refrence,N_ref*D*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_query,N*D*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_distance,section*N_ref*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_temp,section*K*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&gpu_hist,K*N*sizeof(int)));

	
	
	HANDLE_ERROR(cudaMemcpy(gpu_refrence,reference,N_ref*D*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(gpu_query,query,N*D*sizeof(float),cudaMemcpyHostToDevice));
	
	dim3 grid(N_ref/TILEY,section/TILEX,1);
	dim3 block(TILEY,TILEX,1);
	
	int block_dim=K/1024;
	if(K%1024!=0)block_dim++;
	
	
	dim3 block1(thread_x1,thread_y1,1);
	dim3 grid1(block_dim,section,1);
		
	GpuTimer timer;
    timer.Start();
	

	for(int count=0; count<N/section;count++){
		
		
		distance_kernel<<<grid,block>>>(gpu_refrence,gpu_query+count*section*128,gpu_distance);
		
		for(int i=0;i<section;i++){
			
		thrust::device_ptr<float> distance_ptr(gpu_distance+N_ref*i);
                thrust::sort(distance_ptr, distance_ptr + N_ref);
		
		}
			
		binarysearch_kernel<<<grid1,block1>>>(gpu_distance,gpu_temp,K);
		
		sub_kernel<<<grid1,block1>>>(gpu_temp,gpu_hist+section*count*K,K);

		
	}
	
   
  
  	timer.Stop();
	*gpu_kernel_time = timer.Elapsed();

    //Copy to Host and Free the Memory

     HANDLE_ERROR(cudaMemcpy(hist,gpu_hist,N*K*sizeof(float),cudaMemcpyDeviceToHost));
	 
	 HANDLE_ERROR(cudaFree(gpu_refrence));
	 HANDLE_ERROR(cudaFree(gpu_query));
	 HANDLE_ERROR(cudaFree(gpu_temp));
	 HANDLE_ERROR(cudaFree(gpu_distance));
	 HANDLE_ERROR(cudaFree(gpu_hist));
	 
	 for(int i=0;i<200*K;i++){
		 
		 printf("%d    ",hist[i]);
                 if(i%5==4)printf("\n");
                   
	 }


}
//-----------------------------------------------------------------------------
void get_inputs(int argc, char *argv[], unsigned int& N, unsigned int& K)
{
    if (
	argc != 3 ||
	atoi(argv[1]) < 0 || atoi(argv[1]) > 10000 ||
	atoi(argv[2]) < 0 || atoi(argv[2]) > 5000
	) {
        printf("<< Error >>\n");
        printf("Enter the following command:\n");
        printf("\t./nn  N  K\n");
        printf("\t\tN must be between 0 and 10000\n");
        printf("\t\tK must be between 0 and 5000\n");
		exit(-1);
    }
	N = atoi(argv[1]);
	K = atoi(argv[2]);
}
//-----------------------------------------------------------------------------
int fvecs_read (const char *fname, int d, int n, float *a)
{
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}


int ivecs_write (const char *fname, int d, int n, const int *v)
{
  FILE *f = fopen (fname, "w");
  if (!f) {
    perror ("ivecs_write");
    return -1;
  }

  int i;
  for (i = 0 ; i < n ; i++) {
    fwrite (&d, sizeof (d), 1, f);
    fwrite (v, sizeof (*v), d, f);
    v+=d;
  }
  fclose (f);
  return n;
}

