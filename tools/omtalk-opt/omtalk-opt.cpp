// Just make sure that main is referenced so it gets linked in from MLIRMlirOptMain
int main(int argc, char ** argv);

void* dummy_function(){
  return (void*) &main;
}

