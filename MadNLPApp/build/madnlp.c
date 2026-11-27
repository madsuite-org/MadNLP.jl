/* bin/madnlp: the AMPL solver executable, a launcher for libmadnlp's madnlp_main. */
#include <stdint.h>
int madnlp_main(int argc, char** argv);
int main(int argc, char** argv) { return madnlp_main(argc - 1, argv + 1); }
