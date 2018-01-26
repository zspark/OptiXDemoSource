//
// Copyright 1986, 2017 NVIDIA Corporation. All rights reserved.
// program expects one argument, the directory with the beauty .npy files
// it creates links from files in the beauty directory to files in a directory "training"
// which must be present before running the program.

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define MAXFILES 100000

static int numbase;
static char * base[MAXFILES];
static int refsample[MAXFILES];
static int samplecount[MAXFILES][200];
static int numsamples[MAXFILES];

void getbasefiles(char * d)
{
    struct dirent * dr;
    DIR * dirp = opendir(d);
    numbase = 0;
    while ((dr = readdir(dirp))) {
	char tmp[2000];
	struct stat sbuf;
	char * f = strrchr(dr->d_name, '_');
	sprintf(tmp, "%s/%s", d, dr->d_name);
	stat(tmp, &sbuf);
	if (f && S_ISREG(sbuf.st_mode) && access(tmp, F_OK) != -1) {
	    int found = 0, num, i;
	    *f = 0;
	    sscanf(f+1, "%d.npy", &num);
	    for (i=0; i < numbase; i++) {
		if (!strcmp(base[i], dr->d_name)) { /* found */
		    found = 1;
		    if (num > refsample[i])
			refsample[i] = num;
		    samplecount[i][numsamples[i]] = num;
		    numsamples[i] += 1;
		}
	    }
	    if (!found) {
		base[numbase] = (char*)malloc(strlen(dr->d_name)+1);
		strcpy(base[numbase], dr->d_name);
		refsample[numbase] = num;
		numsamples[numbase] = 1;
		samplecount[numbase][0] = num;
		numbase++;
	    }
	}
    }
}

int main(int argc, char ** argv)
{
    int i, k;
    char cmd[2000];

    if (argc != 2) return 0;
    getbasefiles(argv[1]);
    printf("found %d data sets\n", numbase);

    for (i=0; i < numbase; i++) {
	printf("processing data set %s, it has %d sample files, ref samples %d\n", base[i], numsamples[i], refsample[i]);
	sprintf(cmd, "ln %s/%s_%06d.npy training/%s_target.npy", argv[1], base[i], refsample[i], base[i]);
	system(cmd);
	for (k=0; k < numsamples[i]; k++) {
	    if (samplecount[i][k] != refsample[i]) {
		sprintf(cmd, "ln %s/%s_%06d.npy training/%s_%06d.npy",
		    argv[1], base[i], samplecount[i][k], base[i], samplecount[i][k]);
		system(cmd);
	    }
	}
    }
}
