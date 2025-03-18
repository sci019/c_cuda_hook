#include <stdlib.h>

int main()
{
    void *p = malloc(1024);
    free(p);
    return 0;
}