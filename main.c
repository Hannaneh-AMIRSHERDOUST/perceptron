#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define Lines 4    
#define Training_Inputs 3    //avec biais
#define Output 1    
#define DATA_AND "AND-data.txt"  
#define DATA_OR "OR-data.txt" 
#define Training 4  //lignes 
#define Inputs_bias  3 
#define Hidden 4 
#define MIN -0.5
#define MAX 0.5
#define randn() (((double)rand()/((double)RAND_MAX + 1)) * (MAX - MIN)) + MIN
static pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER; // J'ai utilisé mutex pour verrouiller le code afin de pouvoir utiliser les ressources dans les deux fonctions.

float input[4][3];
float weight[4][3];
float targets_and[4];
float targets_or[4];
float Targets_and[4][1]; 
float Targets_or[4][1]; 
float hidden[4][5];
float deltaOutput[1];
float wh[3][4], wz[5][1];
float output[4][1];



void* andFunction(void* arg){
    //ouvrir et lire les données d'entraînement à partir du fichier .txt
    pthread_mutex_lock(&mutex);

    FILE *fpAND;
    fpAND = fopen(DATA_AND, "r");   
    char read_in[150];
    float conv_in;
    int i,k; 
        for (i = 0; i < 4; i++){
            for (k = 0; k <= 3; k++){
                //nous ajoutons un biais au premier élément de chaque ligne dans les données d'entraînement (k==0)
            if (k == 0) {
                input[i][k] = 1;           
            }
            else if (k == 3) {
                                                //nous collectons les cibles ou les sorties souhaitées à partir du fichier .txt (4ème élément ou k==3)
                fscanf(fpAND, "%s", read_in);      //lire et convertir un caractère en entier à virgule flottante
                conv_in = atof(read_in);        
                targets_and[i] = conv_in;
            }
            else {
                fscanf(fpAND, "%s", read_in);      //nous collectons le fichier .txt des entrées (deuxième et troisième éléments ou k==2,k==1)
                conv_in = atof(read_in);        //convertir un caractère en entier à virgule flottante(floating point integer)
                input[i][k] = conv_in;
                }
            }

        }
        fclose(fpAND);

    //lire les entrées et la cible à partir de fichiers .txt et les stocker dans des tableaux terminés.//


    int  p, epoch;
    float error, LearnRate = 0.7;
    float sumHidden[4][4];
    float sumOutput[4][1];
    //nous commençons à initialiser les poids avec la fonction randn().
    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            //poids aléatoires pour l'entrée dans la couche cachée (wh)
            wh[i][k] = randn();  
            printf("\nwh[%d][%d]: %f", i, k, wh[i][k]);
            }
        }

    for (k = 0; k < 1; k++) {
        for (i = 0; i < 3; i++) {
            //poids aléatoires pour la couche cachée à la couche de sortie (wz)
            wz[i][k] = randn();  
            printf("\nwz[%d][%d]: %f", i, k, wz[i][k]);
        }
    }

    for (k = 0; k < 1; k++) {
        for (i = 0; i < 4; i++) {
            //nous convertissons les cibles collectées ou les sorties souhaitées en un tableau à deux dimensions nommé Targets
            Targets_and[i][k] = targets_and[i];
            printf("\n\nTargets[%d][%d]: %f", i, k, Targets_and[i][k]);
        }
    }


    //nous commençons la formation
    for (epoch = 1; epoch <=100; epoch++) {
        error = 0.0;
        for (p = 0; p < 4; p++) { //pour chaque ligne dans les données d'entraînement(p)

            for (k = 0; k < 4; k++) {
                sumHidden[p][k] = 0.0;
                for (i = 0; i < 3; i++) {

                    sumHidden[p][k] = sumHidden[p][k] + input[p][i] * wh[i][k]; //somme des sorties à la couche cachée, biais inclus dans input_data
                }

                if (k == 0) {
                    hidden[p][k] = 1.0; //nous ajoutons un biais au calque caché
                }
                hidden[p][k+1] = 1.0 / (1.0 + exp(-sumHidden[p][k]));    //sorties de la fonction d'activation a la couche cachée

            }
            for (k = 0; k < 1; k++) {
                sumOutput[p][k] = 0.0;
                for (i = 0; i < 5; i++) {
                    sumOutput[p][k] = sumOutput[p][k] + hidden[p][i] * wh[i][k]; //somme des sorties à la couche de sortie
                }

                output[p][k] = 1.0 / (1.0 + exp(-sumOutput[p][k]));    //sortie de la fonction d'activation au niveau de la couche de sortie
                error = error + 0.5 * (Targets_and[p][k] - output[p][k]) * (Targets_and[p][k] - output[p][k]); //calcule erreur
                deltaOutput[k] = (Targets_and[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]);

            }
            printf("\nepoch %-5d :  Error = %f", epoch, error);
        // back propagation learning algorithm
        float sumDOW[5], deltaH[5];
        float deltawh[3][4], deltawz[5][1];


    for (k = 0; k < 1; k++) {
        for (i = 0; i < 5; i++) {
            deltawz[i][k] = 0.0;

        }
    }
    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            deltawh[i][k] = 0.0;

            }
        }

    for (i = 0; i < 5; i++) {
        sumDOW[i] = 0.0;
    }
    for (i = 0; i < 5; i++) {
        deltaH[i] = 0.0;
    }

    //send back errors to hidden layer
    for (i = 0; i < 5; i++)  {
            for (k = 0; k < 5; k++) {

                sumDOW[i] = sumDOW[i] + wz[i][k] * deltaOutput[k];
        }

    }

        for (i = 1; i < 5; i++) {
            deltaH[i] = sumDOW[i] * hidden[p][i] * (1.0 - hidden[p][i]);

        }

    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            deltawh[i][k] = LearnRate * deltaH[k+1] * input[p][i]; 
            wh[i][k] = wh[i][k] + deltawh[i][k]; 
        }

    }
    for (k = 0; k < 1; k++)  {
        for (i = 0; i < 5; i++) {
            deltawz[i][k] = LearnRate * deltaOutput[k] * hidden[p][i]; 
            wz[i][k] = wz[i][k] + deltawz[i][k]; 
        }
    }
        }//end of training

        if (error ==0) { 
            break;
        }
    
    }

    printf("\n\n\tTraining Results AND\n\nPat\t");

    for (i = 1; i < 3; i++) {
        printf("Input%-4d\t", i);
    }
    for (k = 1; k <= Output; k++) {
        printf("Targets\t\tOutputs\t");
    }
    for (p = 0; p < Training; p++) {
        printf("\n%d\t", p);
        for (i = 1; i < 3; i++) {
            printf("%f\t", input[p][i]);
        }
        for(k = 0; k < 1; k++) {
            printf("%f\t%f\t", Targets_and[p][k], output[p][k]);
        }
    }
    pthread_mutex_unlock(&mutex);
    
}








void* orFunction(void* arg){
    pthread_mutex_lock(&mutex);

    FILE *fpOR;
    fpOR = fopen(DATA_OR, "r");   
    char read_OR[150];
    float conv_in;
    int i,k; 
        for (i = 0; i < 4; i++){
            for (k = 0; k <= 3; k++){
            if (k == 0) {
                input[i][k] = 1;           
            }
            else if (k == 3) {
                fscanf(fpOR, "%s", read_OR);     
                conv_in = atof(read_OR);        
                targets_or[i] = conv_in;
            }
            else {
                fscanf(fpOR, "%s", read_OR);      
                conv_in = atof(read_OR);        
                input[i][k] = conv_in;
                }
            }

        }
        fclose(fpOR);

    int  p, epoch;
    float error, LearnRate = 0.7;
    float sumH[4][4];
    float sumO[4][1];

    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            wh[i][k] = randn();  
            printf("\nwh[%d][%d]: %f", i, k, wh[i][k]);
            }
        }

    for (k = 0; k < 1; k++) {
        for (i = 0; i < 3; i++) {
            wz[i][k] = randn();  
            printf("\nwz[%d][%d]: %f", i, k, wz[i][k]);
        }
    }

    for (k = 0; k < 1; k++) {
        for (i = 0; i < 4; i++) {
            Targets_or[i][k] = targets_or[i];
            printf("\n\nTargets[%d][%d]: %f", i, k, Targets_or[i][k]);
        }
    }

    for (epoch = 1; epoch <=100; epoch++) {
        error = 0.0;
        for (p = 0; p < 4; p++) { 

            for (k = 0; k < 4; k++) {
                sumH[p][k] = 0.0;
                for (i = 0; i < 3; i++) {

                    sumH[p][k] = sumH[p][k] + input[p][i] * wh[i][k]; 
                }

                if (k == 0) {
                    hidden[p][k] = 1.0; 
                }
                hidden[p][k+1] = 1.0 / (1.0 + exp(-sumH[p][k]));   

            }
            for (k = 0; k < 1; k++) {
                sumO[p][k] = 0.0;
                for (i = 0; i < 5; i++) {
                    sumO[p][k] = sumO[p][k] + hidden[p][i] * wh[i][k]; 
                }

                output[p][k] = 1.0 / (1.0 + exp(-sumO[p][k]));    
                error = error + 0.5 * (Targets_or[p][k] - output[p][k]) * (Targets_or[p][k] - output[p][k]); 
                deltaOutput[k] = (Targets_or[p][k] - output[p][k]) * output[p][k] * (1.0 - output[p][k]);

            }
            printf("\nepoch %-5d :  Error = %f", epoch, error);
            float sumDOW[5], deltaH[5];
            float deltawh[3][4], deltawz[5][1];


    for (k = 0; k < 1; k++) {
        for (i = 0; i < 5; i++) {
            deltawz[i][k] = 0.0;

        }
    }
    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            deltawh[i][k] = 0.0;

            }
        }

    for (i = 0; i < 5; i++) {
        sumDOW[i] = 0.0;
    }
    for (i = 0; i < 5; i++) {
        deltaH[i] = 0.0;
    }

    for (i = 0; i < 5; i++)  {
            for (k = 0; k < 5; k++) {
                sumDOW[i] = sumDOW[i] + wz[i][k] * deltaOutput[k];
        }

    }

        for (i = 1; i < 5; i++) {
            deltaH[i] = sumDOW[i] * hidden[p][i] * (1.0 - hidden[p][i]);

        }

    for (k = 0; k < 4; k++) {
        for (i = 0; i < 3; i++) {
            deltawh[i][k] = LearnRate * deltaH[k+1] * input[p][i]; 
            wh[i][k] = wh[i][k] + deltawh[i][k]; 
        }

    }
    for (k = 0; k < 1; k++)  {
        for (i = 0; i < 5; i++) {
            deltawz[i][k] = LearnRate * deltaOutput[k] * hidden[p][i]; 
            wz[i][k] = wz[i][k] + deltawz[i][k]; 
        }
    }
        }

        if (error ==0) { 
            break;
        }
    
    }
    
    printf("\n\n\tTraining Results OR\n\nPat\t");

    for (i = 1; i < 3; i++) {
        printf("Input%-4d\t", i);
    }
    for (k = 1; k <= Output; k++) {
        printf("Targets\t\tOutputs\t");
    }
    for (p = 0; p < Training; p++) {
        printf("\n%d\t", p);
        for (i = 1; i < 3; i++) {
            printf("%f\t", input[p][i]);
        }
        for(k = 0; k < 1; k++) {
            printf("%f\t%f\t", Targets_or[p][k], output[p][k]);
        }
    }
    pthread_mutex_unlock(&mutex);
    
}
    

    


int main(){

    pthread_t thread1, thread2;

    // make threads
    pthread_create(&thread1, NULL, andFunction, NULL);
    pthread_create(&thread2, NULL, orFunction, NULL);

    // wait for them to finish if the epoches were different
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL); 

    return 0;
}