#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

struct params
{
    size_t kernel;
    size_t sizeImage;
    size_t stride;
    size_t pading;
    size_t inchannels;
    size_t outchannels;
    double min;
    double max;
};
typedef struct params Params;

struct conv2d
{
    double ***Matrix;
    double ***filters;
    void (*createMemory)(Params p);
    void (*generatedRandomFilter)(Params p);
    void (*convolution)(double ***Matrix, double ***filters,
                        int stride, int pading, int fltNChan, int mtrNChan,
                        int nrowMtr, int nColMtr, int nRowFtr, int nColFtr);
};
typedef struct conv2d Conv2d;

double sigmoid(double x)
{
    return (1 / (1 + exp(x)));
}

double relu(double x)
{
    return x < 0 ? 0 : x;
}

double leakyRelu(double x, int alpha)
{
    return x < 0 ? alpha * x : x;
}

double ***Zeros(double ***Matrix, int rows, int cols, int nchannels)
{
    int i, j, k;
    for (k = 0; k < nchannels; k++)
    {
        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                Matrix[k][i][j] = 0;
            }
        }
    }

    return Matrix;
}

double ***createMemory(double ***Matrix, int nrow, int ncol, int nchannels)
{

    int i, j, k;
    Matrix = malloc(nchannels * sizeof(double **));

    for (k = 0; k < nchannels; k++)
    {
        Matrix[k] = malloc(nrow * sizeof(Matrix[0]));
    }
    for (k = 0; k < nchannels; k++)
    {
        for (i = 0; i < nrow; i++)
        {
            Matrix[k][i] = malloc(ncol * sizeof(Matrix[k][0]));
        }
    }
    Matrix = Zeros(Matrix, nrow, ncol, nchannels);

    return Matrix;
}

void showMatrix(double ***Matrix, int nrow, int ncol, int nchannels)
{
    int i, j, k;
    for (k = 0; k < nchannels; k++)
    { // channels loop
    printf("k == %d \n", k);

        for (i = 0; i < nrow; i++)
        {
            //printf("i == %d ", i);
            for (j = 0; j < ncol; j++)
            {
                printf("  %0.2lf", Matrix[k][i][j]);
            }
            printf("\n");
            printf("\n");
        }
        printf("\n");
        printf("\n");
    }
}

double generateRand(double min, double max)
{

    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void generatedRandomFilter(double ***Matrix, int nrow, int ncol, int nchannels, int min, int max)
{
    int i, j, k;
    srand(time(NULL));

    for (k = 0; k < nchannels; k++)
    {
        for (i = 0; i < nrow; i++)
        {
            for (j = 0; j < ncol; j++)
            {
                Matrix[k][i][j] = generateRand(min, max);
            }
        }
    }
}

double conValue(double ***Matrix, double ***Filter,
                int nColFtr, int nRowFtr, int startRowMtr,
                int startColMtr, int mtrChan, int ftrChan)
{

    int i, j, k, l, m, n;
    double sum = 0;

    for (i = startRowMtr; i < nRowFtr + startRowMtr; i++)
    { //+1
        for (j = startColMtr; j < nColFtr + startColMtr; j++)
        {
            //printf("i == %d",i);

            sum += Matrix[mtrChan][i][j] *
                   Filter[ftrChan][i - startRowMtr][j - startColMtr]; //+ bias
        }
    }
}

void activateChannels(double ***channels, int rows, int cols, int nchan)
{
    int k, l, i;

    for (k = 0; k < nchan; k++)
    { // that a problem
        for (l = 0; l < rows; l++)
        {
            for (i = 0; i < cols; i++)
            {
                channels[k][l][i] = leakyRelu(channels[k][l][i], 0.2);
            }
        }
    }
}

double ***convolution(double ***Matrix, double ***filters,
                      int stride, int pading, int fltNChan, int mtrNChan,
                      int nrowMtr, int nColMtr, int nRowFtr, int nColFtr)
{

    int newftrRow, newftrCol;
    newftrRow = (((nrowMtr - nRowFtr) + 2 * pading) / stride) + 1;
    newftrCol = (((nColMtr - nColFtr) + 2 * pading) / stride) + 1;
    double ***newChannels;
    
    printf("convolution row %d  col %d \n",newftrRow,newftrCol);

    newChannels = createMemory(newChannels, newftrRow, newftrCol, fltNChan);
    
    
    //generatedRandomFilter(filters,newftrRow,newftrCol,newfltrChan,-1,1);
    //showMatrix(newChannels,newftrRow,newftrCol,fltNChan); // change new filter chan
    int i, j, k, l, m, n;
    double sum;
    sum = 0;
    printf("tzzzzzzzzzzzzzzzzz\n");
    
    // i have to do sigmoid for last layer or scalar 
     // that a problem 
    
    // i have to do sigmoid for last layer or scalar
    for (k = 0; k < mtrNChan; k++)
    { // that a problem
        //printf("%d \n",k);
        for (i = 0; i < nrowMtr - nRowFtr+1; i++){

            for (j = 0; j < nColMtr - nColFtr+1; j++){
    
                for (l = 0; l < fltNChan; l++)
                {
                    //printf("%d \n",l);
                //newChannels[k][i][j] =

                    newChannels[k][i][j] += conValue(Matrix, filters, nColFtr,
                                                            nRowFtr, i, j, k, l);
                }

            }
        }
    }
        // printf("\n");
    
    
    //showMatrix(newChannels,newftrRow,newftrCol,newfltrChan);

    //activateChannels(newChannels, newftrRow, newftrCol, fltNChan);
    return newChannels;
}

double*** outPut(double*** inChannels, 
int inrow, int incol, int nchannels, int strid, int padding ){

int outRow, outCol;
double*** outMatrix;
outRow = (incol*strid+padding*2); // make sense
outCol = (incol*strid+padding*2);
outMatrix = createMemory(outMatrix, outRow, outCol, nchannels);
//showMatrix(outMatrix, outRow, outCol, nchannels);


    int k, l, i, frow, fcol;
    

    for (k = 0; k < nchannels; k++)
    {
        frow=0;
        
        for (l = padding; l < outRow-padding; l+=strid)
        {
            fcol=0;
            
            for (i = padding; i < outCol-padding; i+=strid)
            {
                outMatrix[k][l][i] = inChannels[k][frow][fcol];
                fcol++;
            }
            frow++;
        }
    }
    //showMatrix(outMatrix, outRow, outCol, 10); // show first 10
    return outMatrix;
}

double ***deconvolution(double ***Matrix, double ***filters,
                      int stride, int pading, int fltNChan, int mtrNChan,
                      int nrowMtr, int nColMtr, int nRowFtr, int nColFtr)
{

    int newftrRow, newftrCol;
    int ncolout, nrowout;
    newftrRow = nrowMtr*2 + 2*pading;
    newftrCol = nColMtr*2 + 2*pading;

    double*** newChannels;
    double*** out;

    nrowout = newftrRow - nRowFtr + 1;
    ncolout = newftrCol - nColFtr + 1;

    //newftrRow = stride*2 + 2*pading;

    //out = createMemory(out, newftrRow, newftrCol, mtrNChan);
    //generatedRandomFilter(filters,newftrRow,newftrCol,newfltrChan,-1,1);
    //showMatrix(newChannels,newftrRow,newftrCol,fltNChan); // change new filter chan
    
    //out = outPut(Matrix,nrowMtr,nColMtr, fltNChan, 2, 2 );
    printf("row %d  col %d \n",newftrRow,newftrCol);
    printf("nrowout %d  ncolout %d \n",nrowout,ncolout);
    
    newChannels = createMemory(newChannels, nrowout, ncolout, fltNChan);
    //showMatrix(newChannels,nrowout,ncolout,fltNChan);

    
    int i, j, k, l, m, n;
    double sum;
    sum = 0;

    // i have to do sigmoid for last layer or scalar
    for (k = 0; k < fltNChan; k++)
    { // that a problem
        //printf("%d \n",k);
        for (i = 0; i < nrowout; i++){
            for (j = 0; j < ncolout; j++){
    
                for (l = 0; l < mtrNChan; l++)
                {
                    //printf("%d \n",l);
                //newChannels[k][i][j] =

                    newChannels[k][i][j] += conValue(Matrix, filters, nColFtr,
                                                            nRowFtr, i, j, l, k);
                }

            }
        }
    }
    
    //showMatrix(newChannels,newftrRow,newftrCol,newfltrChan);
    activateChannels(newChannels, nrowout, ncolout, fltNChan);
    return newChannels;
}


void main()
{


    double*** Z_distribution_gen1;
    double*** pre_layer1, *** pre_layer2,*** pre_layer3,*** pre_layer4;
    double*** deconv_filter1, *** deconv_filter2;
    double*** deconv_filter3,*** deconv_filter4;

    double*** layer1, ***layer2,***layer3,***layer4;
    

    deconv_filter1 = createMemory(deconv_filter1, 3, 3, 1024);
    generatedRandomFilter(deconv_filter1, 3,3, 1024, -1, 1);

    deconv_filter2 = createMemory(deconv_filter1, 3, 3, 512);
    generatedRandomFilter(deconv_filter2, 3,3, 512, -1, 1);

    deconv_filter3 = createMemory(deconv_filter1, 3, 3, 128);
    generatedRandomFilter(deconv_filter3, 3,3, 128, -1, 1);

    deconv_filter4 = createMemory(deconv_filter1, 3, 3, 3);
    generatedRandomFilter(deconv_filter4, 3, 3, 3, -1, 1);



    Z_distribution_gen1 = createMemory(Z_distribution_gen1, 4,4, 10);
    generatedRandomFilter(Z_distribution_gen1, 4,4, 10, -1, 1);
    
    //showMatrix(Z_distribution_gen1, 4, 4, 2);

    pre_layer1 = outPut(Z_distribution_gen1,4,4,10,2,1);
    //showMatrix(pre_layer1,10 ,10 ,2);


    layer1 = deconvolution(pre_layer1,deconv_filter1,2,1,1024,10,4,4,3,3);
    //showMatrix(layer1, 8,8, 12);


    
    pre_layer2 = outPut(layer1,8,8,1024,2,1);
    //showMatrix(pre_layer2,18,18 ,2);

    layer2 = deconvolution(pre_layer2,deconv_filter2,2,1,512,1024,8,8,3,3);
    //showMatrix(layer2, 16, 16, 2);
    
    


    pre_layer3 = outPut(layer2,16,16,512,2,1);
    //showMatrix(pre_layer3,34 ,34 ,2);

    layer3 = deconvolution(pre_layer3,deconv_filter3,2,1,128,512,16,16,3,3);
   
    //showMatrix(layer3, 32,32, 12);



    pre_layer4 = outPut(layer3,32,32,128,2,1);
    //showMatrix(pre_layer4,66 ,66 ,2);

    layer4 = deconvolution(pre_layer4,deconv_filter4,2,1,3,128,32,32,3,3);
    //showMatrix(layer4, 64,64, 3);


    // descriminator


    double*** desc_pre_layer1, *** desc_pre_layer2,*** desc_pre_layer3,*** desc_pre_layer4;
    double*** desc_conv_filter1, *** desc_conv_filter2;
    double*** desc_conv_filter3,*** desc_conv_filter4;

    double*** desc_layer1, ***desc_layer2,***desc_layer3,***desc_layer4;
    

    desc_conv_filter1 = createMemory(desc_conv_filter1, 3, 3, 64);
    generatedRandomFilter(desc_conv_filter1, 3,3, 64, -1, 1);

    desc_conv_filter2 = createMemory(desc_conv_filter2, 3, 3, 128);
    generatedRandomFilter(desc_conv_filter2, 3,3, 128, -1, 1);

    desc_conv_filter3 = createMemory(desc_conv_filter3, 3, 3, 512);
    generatedRandomFilter(desc_conv_filter3, 3,3, 512, -1, 1);

    desc_conv_filter4 = createMemory(desc_conv_filter4, 3, 3, 1024 );
    generatedRandomFilter(desc_conv_filter4, 3, 3, 1024, -1, 1);


    //desc_layer1 = createMemory(desc_layer1, 64, 64, 64);
    //desc_layer1 = convolution(layer4,desc_conv_filter1,1,0,64,3,64,64,3,3);
    
    
    desc_layer1 = convolution(layer4,desc_conv_filter1,1,1,64,3,64,64,3,3);
    //showMatrix(Z_distribution_gen1, 4, 4, 2);
    
    desc_layer2 = convolution(desc_layer1,desc_conv_filter2,1,1,128,64,64,64,3,3);
    //showMatrix(Z_distribution_gen1, 4, 4, 2);

    desc_layer3 = convolution(desc_layer2,desc_conv_filter3,1,1,512,128,64,64,3,3);
    //showMatrix(Z_distribution_gen1, 4, 4, 2);


    desc_layer4 = convolution(desc_layer3,desc_conv_filter4,1,1,1024,512,64,64,3,3);
    //showMatrix(Z_distribution_gen1, 4, 4, 2);


    

}