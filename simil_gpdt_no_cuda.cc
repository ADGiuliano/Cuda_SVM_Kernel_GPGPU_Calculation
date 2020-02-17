/**
 * Programma in versione seriale che simula il 
 * comportamento del gpdt per la risoluzione  
 * di un certo tipo di kernel di una serie di valori di 
 * dimensione variabile.
 * compilare con:
 * g++ -o simil_gpdt_no_cuda simil_gpdt_no_cuda.cc
 * lanciare con:
 * ./simil_gpdt_no_cuda [numero vettori] [numero componenti] [numero di righe da calcolare] [tipo di kernel] [grado(int)/sigma(float)]
 **/
 
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <math.h>
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
 * Funzione per il calcolo delle norme al quadrato.
 **/
void calcolo_norme(float *vett, float *norme, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val)
{
	int pos;
	int Nr_val;
	float norma;
	for(int i = 0; i < N; i++)
	{
		norma = 0;
		
		Nr_val = vett_nr_posizioni[i];
		
		for(int j = 0; j < Nr_val; j++)
		{
			pos = vett_posizioni[ i * nr_max_val +j];
			
			norma = norma + (vett[i * C + pos] * vett[i * C + pos]);
		}
		
		norme[i] = norma;
	}
}

/**
 * Funzione per il calcolo del kernel lineare
 **/
void calcolo_linear(float *vett, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultati, int *indici, int num_indici)
{
	int pos;
	int tmp_ind;
	int Nr_val;
	float lin;
	
	for(int i=0; i< N; i++)
	{
		for(int j=0; j<num_indici; j++)
		{
			tmp_ind = indici[j];
			
			lin = 0.0;
			
			Nr_val = vett_nr_posizioni[i];
			
			for(int kk=0; kk < Nr_val; kk++)
			{
				pos = vett_posizioni[ i * nr_max_val +kk];
				lin = lin + (vett[i * C + pos] * vett[tmp_ind * C + pos]);
			}
			
			//risultati[i*num_indici+j] = lin;
			risultati[j*N+i] = lin;
		}
	}
}

/**
 * Funzione per il calcolo del kernel polimoniale.
 **/
void calcolo_pol(float *vett, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultati, int *indici, int num_indici, int s)
{
	int pos;
	int tmp_ind;
	int Nr_val;
	float pol;
	float tmp;
	
	for(int i=0; i< N; i++)
	{
		for(int j=0; j<num_indici; j++)
		{
			tmp_ind = indici[j];
			
			pol = 0.0;
			
			tmp = 1.0;
			
			Nr_val = vett_nr_posizioni[i];
			
			for(int kk=0; kk < Nr_val; kk++)
			{
				pos = vett_posizioni[ i * nr_max_val +kk];
				pol = pol + (vett[i * C + pos] * vett[tmp_ind * C + pos]);
			}
			
			pol = pol + 1;
			
			for (int a = 0; a < s ; a++)
			{
				tmp = tmp * pol;
			}
			
			//risultati[i*num_indici+j] = tmp;
			risultati[j*N+i] = tmp;
		}
	}
}


/**
 * Funzione per il calcolo del kernel gaussiano.
 **/
void calcolo_gaus(float *vett, float *norme, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultati, float sigma, int *indici, int num_indici)
{
	int pos;
	int tmp_ind;
	int Nr_val;
	float gaus;
	
	for(int i=0; i< N; i++)
	{
		for(int j=0; j<num_indici; j++)
		{
			tmp_ind = indici[j];
			
			gaus = 0.0;
			
			Nr_val = vett_nr_posizioni[i];
			
			for(int kk=0; kk < Nr_val; kk++)
			{
				pos = vett_posizioni[ i * nr_max_val +kk];
				gaus = gaus + (vett[i * C + pos] * vett[tmp_ind * C + pos]);
			}
			
			gaus = -2.0*gaus + norme[i] + norme[tmp_ind];
			
			gaus = (exp(-gaus*sigma));
			
			//risultati[i*num_indici+j] = gaus;
			risultati[j*N+i] = gaus;
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
	 * vettore contenente le norme 2 al quadrato dei vettori.
	 **/
	float *vett_norme;
	
	/**
	 * Numero di vettori.
	 **/
	int Nr_vet_elem = atoi(argv[1]);
	
	/**
	 * Numero di elementi per vettore.
	 **/
	int Nr_vet_comp = atoi(argv[2]);
	
	/**
	 * Numero di righe da calcolare.
	 **/
	int Nr_righe_calc = atoi(argv[3]);
	
	/**
	 * Numero per la selezione del kernel.
	 * 1 = kernel lineare.
	 * 2 = kernel polimoniale.
	 * 3 = kernel gaussiano.
	 **/
	int sel_kernel = atoi(argv[4]);
	
	/**
	 * Matrice contenente i risultati finali.
	 **/
	float *risultati;
	
	/**
	 * Vettore contenente gli indici da calcolare.
	 **/
	int *indici;
	
	/**
	 * Variabile contenente il numero dei valori significativi.
	 **/
	int numero_val_significativi = Nr_vet_comp * perc_val_noti;
	
	//dati per il conteggio temporale del programma.
	clock_t startTime, stopTime;
	double totTime;
	
	if(Nr_righe_calc > Nr_vet_elem)
	{
		Nr_righe_calc = Nr_vet_elem;
	}
	
	//Spazio necessario per l'allocazione dei vettori.
	int tot_vett_size = Nr_vet_elem * Nr_vet_comp * sizeof(float);
	//Spazio necessario per l'allocazione della Matrice dei risultati.
	int norme_size = Nr_vet_elem * sizeof(float);
	//Spazio necessario per l'allocazione della Matrice delle posizioni.
	int vett_pos_size = Nr_vet_elem * numero_val_significativi * sizeof(int);
	//Spazio necessario per l'allocazione del vettore con il numero dei valori significativi.
	int vett_nrpos_size = Nr_vet_elem * sizeof(int);
	//Spazio necessario per l'allocazione della matrice dei risultati finali.
	int risultati_size = Nr_vet_elem * Nr_righe_calc * sizeof(float);
	//Spazio necessario per l'allocazione del vettore degli indici.
	int indici_size = Nr_righe_calc * sizeof(int);
	
	//Allocazione.
	vettori = (float*)malloc(tot_vett_size);
	vett_norme = (float*)malloc(norme_size);
	vettore_posizioni = (int*)malloc(vett_pos_size);
	vett_numero_posizioni = (int*)malloc(vett_nrpos_size);
	risultati = (float*)malloc(risultati_size);
	indici = (int*)malloc(indici_size);
	
	//Riempimento dei vettori.
	riempi_vettori(vettori, Nr_vet_elem, Nr_vet_comp);
	//Riempimento dei vettori delle posizioni.
	crea_vettori_termini_noti(vettore_posizioni, Nr_vet_elem, numero_val_significativi);
	//Riempimento del vettore contenente il numero dei valori significativi.
	crea_vettori_posizioni(vett_numero_posizioni, Nr_vet_elem, numero_val_significativi);
	
	int contatore = 0;
	
	for(int i=0; i < Nr_righe_calc; i++)
	{
		indici[i] = contatore + i;
	}
	
	switch(sel_kernel){
			
			case(1):{
				//cout<<"Kernel Lineare\n";
				
				startTime = clock();
				//void calcolo_linear(float *vett, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultati, int *indici, int num_indici)
				calcolo_linear(vettori, vettore_posizioni, vett_numero_posizioni, Nr_vet_elem, Nr_vet_comp, numero_val_significativi, risultati, indici, Nr_righe_calc);
				stopTime = clock();
				totTime = (double) (stopTime - startTime) / CLOCKS_PER_SEC;
				break;
			}
			case(2):{
				//cout<<"Kernel Polimoniale\n";
				/**
				 * Grado del kernel.
				 **/
				int grado = atoi(argv[5]);
				
				startTime = clock();
				//void calcolo_pol(float *vett, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultati, int *indici, int num_indici, int s)
				calcolo_pol(vettori, vettore_posizioni, vett_numero_posizioni, Nr_vet_elem, Nr_vet_comp, numero_val_significativi, risultati, indici, Nr_righe_calc, grado);
				stopTime = clock();
				totTime = (double) (stopTime - startTime) / CLOCKS_PER_SEC;
				break;
			}
			
			case(3):{
				//cout<<"Kernel gaussiano\n";
				
				/**
				* Sigma della funzione gaussiana.
				**/
				float sigma = atof(argv[5]);
				sigma = (1.0/(2.0*sigma*sigma));
				
				startTime = clock();
				
				//calcolo norme al quadrato
				calcolo_norme(vettori, vett_norme, vettore_posizioni, vett_numero_posizioni, Nr_vet_elem, Nr_vet_comp, numero_val_significativi);
				
				//void calcolo_gaus(float *vett, float *norme, int *vett_posizioni, int *vett_nr_posizioni , int N, int C, int nr_max_val, float *risultato, float *sigma, int *indici, int num_indici)
				calcolo_gaus(vettori,vett_norme, vettore_posizioni, vett_numero_posizioni, Nr_vet_elem, Nr_vet_comp, numero_val_significativi, risultati, sigma, indici, Nr_righe_calc);
				
				stopTime = clock();
				totTime = (double) (stopTime - startTime) / CLOCKS_PER_SEC;
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
	
	cout<<"Tempo totale:\t"<<totTime<<" secondi\n";
	
	/*for (int i = 0; i <  Nr_vet_elem*Nr_righe_calc; i++)
	{
	  cout<<risultati[i]<<endl;
	}*/
	
	free(vettori);
	free(vett_norme);
	free(vettore_posizioni);
	free(vett_numero_posizioni);
	free(risultati);
	free(indici);
	
	return 0;
}
