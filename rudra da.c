#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SIZE 25
#define SAMPLE 100

double RandomNumber()
{
    return ((double)rand() / (1.0 + RAND_MAX));
}

double MLE_Mean(double *sample)
{
    double sum = 0;
    for (int i = 0; i < SIZE; i++)
        sum += sample[i];
    return sum / SIZE;
}

double MLE_Variance(double *sample, double mean)
{
    double sum = 0;
    for (int i = 0; i < SIZE; i++)
        sum += pow(sample[i] - mean, 2);
    return sum / SIZE;
}

void printSample(double *sample)
{
    for (int i = 0; i < SIZE; i++)
    {
        printf("%lf\n", sample[i]);
    }
}

int main()
{
    srand(time(0));
    double mean = -5;
    double variance = 0.5;
    int i, j;

    // Generating Random Sample //
    double samples[SAMPLE][SIZE];
    for (i = 0; i < SAMPLE; i++)
    {
        for (j = 0; j < SIZE; j++)
            samples[i][j] = mean + sqrt(variance) * RandomNumber();
    }

    double mle_mean[SAMPLE], mle_var[SAMPLE];
    for (i = 0; i < SAMPLE; i++)
    {
        mle_mean[i] = MLE_Mean(samples[i]);
        mle_var[i] = MLE_Variance(samples[i], mle_mean[i]);
    }

    int chosen_sample = rand() % SAMPLE; // Corrected this line

    printSample(samples[chosen_sample]);
    printf("\nMLE Mean: %lf", mle_mean[chosen_sample]);
    printf("\nMLE Variance: %lf", mle_var[chosen_sample]);

    return 0;
}
