#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage() {
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x) {
  // one character per one bit
  char output[32 + 1] = {
      0
  };

  /* YOUR CODE START HERE */
  int len = 32;
  for(int shift=0; shift < len ; shift++){
    
    output[len-1-shift] = (x >> shift) & 1;
    // printf("%c\n", (x >> shift) % 2);
  };

  output[len] = '\0';

  for(int i=0; i < len; i++){
    output[i] = output[i] + '0';
  }
  
  // 1000 0000 0000 0000 0000 0000 0011 11111
  // 10000000000000000000000000111111
  // 00000000000000000000000001111111

  /* YOUR CODE END HERE */
  printf("%s\n", output);
}

void print_long(long x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */

  int len = 64;
  for(int shift=0; shift < len ; shift++){
    
    output[len-1-shift] = (x >> shift) & 1;
    // printf("%c\n", (x >> shift) % 2);
  };

  output[len] = '\0';

  for(int i=0; i < len; i++){
    output[i] = output[i] + '0';
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */

  int len = 32;
  int convert_float = *(unsigned int*)&x;
  for(int shift=0; shift < len ; shift++){
    
    output[len-1-shift] = (convert_float >> shift) & 1;
    // printf("%c\n", (x >> shift) % 2);
  };


  output[len] = '\0';

  for(int i=0; i < len; i++){
    output[i] = output[i] + '0';
  }

  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int len = 64;
  long int  convert_float = *(unsigned long long*)&x;
  for(int shift=0; shift < len ; shift++){
    
    output[len-1-shift] = (convert_float >> shift) & 1;
    // printf("%c\n", (x >> shift) % 2);
  };

  output[len] = '\0';

  for(int i=0; i < len; i++){
    output[i] = output[i] + '0';
  }  

  /* YOUR CODE END HERE */
  printf("%s\n", output);
}

int main(int argc, char **argv) {
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0) {
    print_int(atoi(argv[2]));
  } else if (strcmp(argv[1], "long") == 0) {
    print_long(atol(argv[2]));
  } else if (strcmp(argv[1], "float") == 0) {
    print_float(atof(argv[2]));
  } else if (strcmp(argv[1], "double") == 0) {
    print_double(atof(argv[2]));
  } else {
    fallback_print_usage();
  }
}
