# napkin

Simple sheet of fabric simulated using mass spring system

threads are used to spread the computation over all accessible nodes.

Use glut in an simple and very old manner

mouse drag allow some change of point of view, + / - to zoom

gravity is thrown to the top, just for fun

compilation is trivial:

g++ -march=native -O3 main.cpp -lglut -lGL -pthread  -o napkin 
