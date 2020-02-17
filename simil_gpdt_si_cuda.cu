/**
 * Programma che simula il comportamento del gpdt per 
 * la risoluzione di un kernel di una serie di
 * valori di dimensione variabile utilizzando la 
 * tecnologia cuda.
 * compilare con:
 * nvcc -o simil_gpdt_si_cuda simil_gpdt_si_cuda.cu
 * lanciare con:
 * ./simil_gpdt_si_cuda [numero vettori] [numero componenti] [numero di righe da calcolare] [tipo di kernel] [grado(int)/sigma(float)]
 **/

#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <cuda.h>
using namespace std;

/**
 * Funzione che riempie i vettori con numeri
 * casuali compresi tra 0 e 99.
 **/ 
void riempi_vettori(float *vettori, int Nr_vet_elem, int Nr_vet_comp)
{
	for (int i = 0; i < Nr_vet_elem; i++)
		for(int j = 0; j < Nr_vet_comp; j++)
			vettori[i * Nr_vet_comp + j] = i * 2 + j; //j % 4; //
}

/**
 * Funzione che crea dei vettori contenente i valori significativi su cui 
 * calcolare la norma 2 al quadrato.
 **/
void crea_vettori_termini_noti(int *vettori, int Nr_vet_elem, int Nr_vet_comp)
{
	for (int i = 0; i < Nr_vet_elem; i++)
		for(int j = 0; j < Nr_vet_comp; j++)
			vettori[i * Nr_vet_comp + j] = (j+1)*3;
}

/**
 * Funzione che crea un vettore contenente il numero di valori significative.
 **/
void crea_vettori_posizioni(int *vettore, int Nr_vet_elem, int numero_val)
{
	for (int i = 0; i < Nr_vet_elem; i++)
		vettore[i] = numero_val;
}

/**
 * kernel per il calcolo delle norme al quadrato dei vettori.
 **/
__global__ void Kernel_norme(float *Vd, float *Nd, int *Vp, int *Vnp, int N, int C, int nr_max_val)
{
	long int x = threadIdx.x + blockIdx.x * blockDim.x;
	
	int pos;
	
	if(x < N)
	{
		float norma = 0;
		
		int Nr_val = Vnp[x];
		
		for(int i = 0; i < Nr_val; i++)
		{
			pos = Vp[x * nr_max_val + i];
			norma = norma + (Vd[x * C + pos] * Vd[x * C + pos]);
		}
		
		Nd[x] = norma;
	}
	
} 

/**
 * Kernel per il calcolo del del guassiano, basato sul metodo utilizzato nel gpdt, 
 * modificato per l'utilizzo con la tecnologia CUDA.
 **/
__global__ void Kernel_gaus(float *Vd, float *Ris, float *Nd, int N, int C, int dim_indici, int *ind, float sigma, int *Vp, int *Vnp, int nr_max_val)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int j;
	int pos;
	int tmp_ind;
	float gaus;
	
	for ( ; x < N ; x+=blockDim.x * gridDim.x)
	{
	    for( ; y < dim_indici; y+=blockDim.y * gridDim.y)
	    {
		      tmp_ind = ind[y];
		    
		      gaus = 0.0;
		      
		      int Nr_val = Vnp[x];
		      
		      for(j = 0; j < Nr_val; j++)
		      {
			      pos = Vp[x * nr_max_val + j];
			      gaus = gaus + (Vd[x * C + pos] * Vd[tmp_ind * C + pos]);
		      }
		      
		      gaus = - 2.0*gaus +Nd[x] + Nd[tmp_ind];
		      gaus = (exp(-gaus*sigma));

		      //Ris[x * dim_indici + y] = gaus;
		      Ris[y * N + x] = gaus;
	    }
	}
}

/**
 * Kernel per il calcolo del kernel lineare 
 * modificato per l'utilizzo con la tecnologia CUDA.
 **/
__global__ void Kernel_lineare(float *Vd, float *Ris, int N, int C, int dim_indici, int *ind, int *Vp, int *Vnp, int nr_max_val)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int j;
	int pos;
	int tmp_ind;
	float lin;
	
	for ( ; x < N ; x+=blockDim.x * gridDim.x)
	{
	    for( ; y < dim_indici; y+=blockDim.y * gridDim.y)
	    {
		      tmp_ind = ind[y];
		    
		      lin = 0.0;
		      
		      int Nr_val = Vnp[x];
		      
		      for(j = 0; j < Nr_val; j++)
		      {
			      pos = Vp[x * nr_max_val + j];
			      lin =  lin + (Vd[x * C + pos] * Vd[tmp_ind * C + pos]);
		      }

		      //Ris[x * dim_indici + y] = lin;
		      Ris[y * N + x ] = lin;
	    }
	}
}


/**
 * Kernel per il calcolo del kernel lineare 
 * modificato per l'utilizzo con la tecnologia CUDA.
 **/
__global__ void Kernel_polimoniale(float *Vd, float *Ris, int N, int C, int dim_indici, int *ind, int *Vp, int *Vnp, int nr_max_val, int s)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int j;
	int pos;
	int tmp_ind;
	float pol;
	float tmp;
	
	for ( ; x < N ; x+=blockDim.x * gridDim.x)
	{
	    for( ; y < dim_indici; y+=blockDim.y * gridDim.y)
	    {
		      tmp_ind = ind[y];
			  
			  tmp = 1.0;
			  
		      pol = 0.0;
		      
		      int Nr_val = Vnp[x];
		      
		      for(j = 0; j < Nr_val; j++)
		      {
			      pos = Vp[x * nr_max_val + j];
			      pol = pol + (Vd[x * C + pos] * Vd[tmp_ind * C + pos]);
		      }
		      
		      pol = pol + 1;
		      
		      for(j = 0; j < s; j++)
		      {
				tmp = tmp * pol;
			  }

		      //Ris[x * dim_indici + y] = tmp;
		      Ris[y * N + x ] = tmp;
	    }
	}
}



int main(int argc, char** argv)

{	
	/**
	 * Variabile contenente la percentuale dei valori significativi 
	 * all'interno dei vettori.
	 **/
	float perc_val_noti= 1.0 - 0.82;
	
	/**
	 * Matrice contenente i vettori da cui calcolare la differeza 
	 * per calcolarne le norme.
	 * Il numero di vettori e la dimensione degli stessi viene definita
	 * dall'utente.
	 **/
	float *vettori;
	
	/**
	 * Matrice contenente le posizioni all'interno del vettore contenente
	 * le posizioni dei valori significativi.
	 **/
	 int *vettore_posizioni;
	
	/**
	 * Vettore contenente il numero di valori non nulli nel vettore.
	 **/
	int *vett_numero_posizioni;
	
	/**
	 * vettore contenente le norme 2 al quadreato dei vettori.
	 **/
	float *vett_norme;
	
	/**
	 * Matrice contenente i risultati.
	 **/
	float *risultati;
	
	/**
	 * 
	 **/
	int *indici; 
	
	/**
	 * Tempo impiegato per il calcolo
	 **/
	float elapsedTime;
	cudaEvent_t start, stop;
	
	//nr di vettori e di elementi.
	int Nr_vet_elem = atoi(argv[1]);
	int Nr_vet_comp = atoi(argv[2]);
	
	//Numero di righe da calcolare.
	int Nr_righe = atoi(argv[3]);
	
	/**
	 * Numero per la selezione del kernel.
	 * 1 = kernel lineare.
	 * 2 = kernel polimoniale.
	 * 3 = kernel gaussiano.
	 **/
	int sel_kernel = atoi(argv[4]);
	
	/**
	 * Sigma della funzione gaussiana.
	 **/
	//float sigma = atoi(argv[5]);
	//sigma = (1.0/(2.0*sigma*sigma));
	
	//Copia per il device.
	float *Vd;
	int *Vp;
	int *Vnp;
	float *Nd;
	float *Ris;
	int *ind;
	
	//Variabili per il controllo della memoria disponibile.
	size_t free_byte;
	size_t total_byte;
	
	/**
	 * Variabile contenente il numero dei valori significativi.
	 **/
	int numero_val_significativi = Nr_vet_comp * perc_val_noti;
	
	//Spazio necessario per l'allocazione dei vettori.
	int tot_vett_size = Nr_vet_elem * Nr_vet_comp * sizeof(float);
	//Spazio necessario per l'allocazione della Matrice dei risultati.
	int norme_size = Nr_vet_elem * sizeof(float);
	//Spazio necessario per l'allocazione della Matrice delle posizioni.
	int vett_pos_size = Nr_vet_elem * numero_val_significativi * sizeof(int);
	//Spazio necessario per l'allocazione del vettore con il numero dei valori significativi.
	int vett_nrpos_size = Nr_vet_elem * sizeof(int);
	//Spazio necessario per l'allocazione di una colonna.
	int col_size = Nr_vet_elem * sizeof(float);
	
	
	
	//Allocazione.
	vettori = (float*)malloc(tot_vett_size);
	vett_norme = (float*)malloc(norme_size);
	vettore_posizioni = (int*)malloc(vett_pos_size);
	vett_numero_posizioni = (int*)malloc(vett_nrpos_size);
	
	
	
	//Allocazione nel device.
	cudaMalloc((void **)&Vd, tot_vett_size);
	cudaMalloc((void **)&Nd, norme_size);
	cudaMalloc((void **)&Vp, vett_pos_size);
	cudaMalloc((void **)&Vnp, vett_nrpos_size);
	
	srand(time(0));
	
	//Riempimento dei vettori.
	riempi_vettori(vettori, Nr_vet_elem, Nr_vet_comp);
	//Riempimento dei vettori delle posizioni.
	crea_vettori_termini_noti(vettore_posizioni, Nr_vet_elem, numero_val_significativi);
	//Riempimento del vettore contenente il numero dei valori significativi.
	crea_vettori_posizioni(vett_numero_posizioni, Nr_vet_elem, numero_val_significativi);
		
	//trasferimento dei vettori nel device.
	cudaMemcpy(Vd, vettori, tot_vett_size, cudaMemcpyHostToDevice);
	//trasferimento dei vettori delle posizioni nel device.
	cudaMemcpy(Vp, vettore_posizioni, vett_pos_size, cudaMemcpyHostToDevice);
	//trasferimento del vettore conentente il numero di valori all'interno di ogni singolo vettore.
	cudaMemcpy(Vnp, vett_numero_posizioni, vett_nrpos_size, cudaMemcpyHostToDevice);
	
	cudaMemGetInfo( &free_byte, &total_byte );
	
	int col_ospitabili_mem = (free_byte*0.7)/col_size;
	
	int contatore = 0;
	
	/**
	 * Valori impostati per ottimizzare il funzionamento del device.
	 * Questi valori sono basati sull'utilizzo di una Nvidia 230m.
	 **/
	int dimXX =4;
	int dimYY =128;
	
	/**
	 * Numero di colonne ospitabili calcolabili dal kernel contemporaneamente.
	 * Purtroppo a causa del fatto che il kernel CUDA fallisca in automatico
	 * se impiega più di 5 secondi per il calcolo, è necessario inserire un 
	 * limitatore per il calcolo.
	 * Questo valore è basato sull'utilizzo di una Nvidia 230m.
	 **/
	 
	
	int col_ospitabili = 200;
	
	if (col_ospitabili > Nr_righe)
	{
		col_ospitabili = Nr_righe;
	} 
	
	if (col_ospitabili > col_ospitabili_mem)
	{
		col_ospitabili = col_ospitabili_mem;
	}
	
	int numero_cicli = Nr_righe/col_ospitabili;
	cout<<"Numero cicli necessari: "<<numero_cicli<<endl;
	
	int risultati_size = Nr_righe * Nr_vet_elem * sizeof(float);
	int indici_size = col_ospitabili * sizeof(int);
	
	int risultati_part_size = col_ospitabili * Nr_vet_elem * sizeof(float);
	
	risultati = (float*)malloc(risultati_size);
	indici = (int*)malloc(indici_size);
	
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	
	cudaMalloc((void **)&Ris, risultati_part_size);
	cudaMalloc((void **)&ind, indici_size);
	
	dim3 blockGridRows;
	
	blockGridRows.x=Nr_vet_elem/dimXX + (Nr_vet_elem%dimXX== 0?0:1);;
	blockGridRows.y=col_ospitabili/dimYY + (col_ospitabili%dimYY== 0?0:1);
	
	dim3 threadBlockRows;
	threadBlockRows.x=dimXX;
	threadBlockRows.y=dimYY;
	
	cout<<"Memoria allocata, griglie definite:"<<endl;
	cout<<"blockGridRows.x: "<<blockGridRows.x<<endl;
	cout<<"blockGridRows.y: "<<blockGridRows.y<<endl;
	cout<<"threadBlockRows.x: "<<threadBlockRows.x<<endl;
	cout<<"threadBlockRows.y: "<<threadBlockRows.y<<endl;
	
	int cicle_dim = col_ospitabili * Nr_vet_elem * sizeof(float);
	
	dim3 blockGridRowsn(Nr_vet_elem, 1);
	
	int resto;
	
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	switch(sel_kernel){
			case(1):{
				//cout<<"Kernel Lineare\n";
							
				for(int i = 0; i < numero_cicli; i++)
				{
					for (int kk = 0; kk < col_ospitabili; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					//__global__ void Kernel_lineare(float *Vd, float *Ris, int N, int C, int dim_indici, int *ind, int *Vp, int *Vnp, int nr_max_val)
					Kernel_lineare<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nr_vet_elem, Nr_vet_comp, col_ospitabili, ind, Vp, Vnp, numero_val_significativi);
					
					cudaMemcpy(risultati+(i*col_ospitabili*Nr_vet_elem), Ris, cicle_dim, cudaMemcpyDeviceToHost);
					
					contatore = contatore + col_ospitabili;
				}
				
				resto = Nr_righe - contatore;
				
				if (resto > 0)
				{
					for (int kk = 0; kk < resto; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					Kernel_lineare<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nr_vet_elem, Nr_vet_comp, resto, ind, Vp, Vnp, numero_val_significativi);
					
					cudaMemcpy(risultati+(numero_cicli)*(col_ospitabili*Nr_vet_elem), Ris, resto * Nr_vet_elem * sizeof(float), cudaMemcpyDeviceToHost);
					
				}
				
				break;
			}
			
			case(2):{
				//cout<<"Kernel Polimoniale\n";
				/**
				 * Grado del kernel.
				 **/
				int grado = atoi(argv[5]);
				
				for(int i = 0; i < numero_cicli; i++)
				{
					for (int kk = 0; kk < col_ospitabili; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					//__global__ void Kernel_polimoniale(float *Vd, float *Ris, int N, int C, int dim_indici, int *ind, int *Vp, int *Vnp, int nr_max_val, int s)
					Kernel_polimoniale<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nr_vet_elem, Nr_vet_comp, col_ospitabili, ind, Vp, Vnp, numero_val_significativi,grado);
					
					cudaMemcpy(risultati+(i*col_ospitabili*Nr_vet_elem), Ris, cicle_dim, cudaMemcpyDeviceToHost);
					
					contatore = contatore + col_ospitabili;
				}
				
				resto = Nr_righe - contatore;
				
				if (resto > 0)
				{
					for (int kk = 0; kk < resto; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					Kernel_polimoniale<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nr_vet_elem, Nr_vet_comp, resto, ind, Vp, Vnp, numero_val_significativi,grado);
					
					cudaMemcpy(risultati+(numero_cicli)*(col_ospitabili*Nr_vet_elem), Ris, resto * Nr_vet_elem * sizeof(float), cudaMemcpyDeviceToHost);
					
				}
				
				break;
			}
			
			case(3):{
				//cout<<"Kernel gaussiano\n";
				
				/**
				* Sigma della funzione gaussiana.
				**/
				float sigma = atof(argv[5]);
				sigma = (1.0/(2.0*sigma*sigma));
				
				//calcolo norme.
				Kernel_norme<<< blockGridRowsn, 256 >>>(Vd, Nd, Vp, Vnp, Nr_vet_elem, Nr_vet_comp, numero_val_significativi);
				
				//calcolo kernel
				for(int i = 0; i < numero_cicli; i++)
				{
					for (int kk = 0; kk < col_ospitabili; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					//Kernel_gaus(float *Vd, float *Ris, float *Nd, int N, int C, int dim_indici, int *ind, float sigma, float *Vp, float *Vnp)
					Kernel_gaus<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nd, Nr_vet_elem, Nr_vet_comp, col_ospitabili, ind, sigma, Vp, Vnp, numero_val_significativi);
					
					cudaMemcpy(risultati+(i*col_ospitabili*Nr_vet_elem), Ris, cicle_dim, cudaMemcpyDeviceToHost);
					
					contatore = contatore + col_ospitabili;
				}
				
				resto = Nr_righe - contatore;
				
				if (resto > 0)
				{
					for (int kk = 0; kk < resto; kk++)
					{
						indici[kk] = contatore + kk;
					}
					
					cudaMemcpy(ind, indici, indici_size, cudaMemcpyHostToDevice);
					
					Kernel_gaus<<< blockGridRows, threadBlockRows>>>(Vd, Ris, Nd, Nr_vet_elem, Nr_vet_comp, resto, ind, sigma, Vp, Vnp, numero_val_significativi);

					
					cudaMemcpy(risultati+(numero_cicli)*(col_ospitabili*Nr_vet_elem), Ris, resto * Nr_vet_elem * sizeof(float), cudaMemcpyDeviceToHost);
					
				}
				
				break;
			}
			
			default:
			{
			cout<<"Scelta non valida.\n";
			cout<<"4° argomento non esistente.\n";
			cout<<"1 = kernel lineare\t2 = kernel polimoniale\t 3 = kernel gaussiano\n";
			break;
			}
			
			
			
	}
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&elapsedTime, start,stop);
	cout<<"Tempo totale:\t"<<elapsedTime/1000<<" secondi\n";
	
	/*for (int i = 0; i <  Nr_vet_elem*Nr_righe; i++)
	{
	  cout<<risultati[i]<<endl;
	}*/
	
	free(vettori);
	free(vett_norme);
	free(vettore_posizioni);
	free(vett_numero_posizioni);
	free(indici);
	free(risultati);
	
	cudaFree(Vd);
	cudaFree(Vp);
	cudaFree(Vnp);
	cudaFree(Nd);
	cudaFree(ind);
	cudaFree(Ris);
	
	return 0;
}
