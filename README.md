# A Unified Particle Physics Engine
A CUDA + OpenGL powered position-based physics engine.

## Description
This framework supports fluid, clothes and solid.
This is possible due to unified particle physics [1].
The unified particle physics combines the position-based approaches from several papers:
1. Position-based dynamics [4]
2. Position-based fluids [5]
3. Shape-matching [6]
4. To make the shape-matching stable, the robust rotation extraction method [7] is implemented
5. The surface tension method [2]
6. Aerodynamics for clothes and flag[3]
7. For efficient collision detection and density estimation, an efficient hash-based searching method [8] is used.

## What this framework is capable of
[![unified particle physics](http://jamorn.me/pics/upp_large..jpg)](https://www.youtube.com/watch?v=-DgD_PovEdk)

https://www.youtube.com/watch?v=-DgD_PovEdk

## NOTE:
Sadly, There are some bugs to be killed. (In multi-phase fluid test, the denser fluid floats.) There are a lot of duplicated code to be removed. :(

References:
1. MACKLIN , M., MÜLLER , M., CHENTANEZ , N., AND KIM , T.-Y. 2014. Unified particle physics for real-time applications. ACM Transactions on Graphics (TOG) 33, 4, 153.
2. AKINCI , N., AKINCI , G., AND TESCHNER , M. 2013. Versatile surface tension and adhesion for sph fluids. ACM Transactions
on Graphics (TOG) 32, 6, 182.
3. KECKEISEN , M., KIMMERLE , S., THOMASZEWSKI , B., AND W ACKER , M. 2004. Modelling effects of wind fields in cloth
animations.
4. MÜLLER , M., HEIDELBERGER , B., HENNIX , M., AND RATCLIFF, J. 2007. Position based dynamics. Journal of Visual
Communication and Image Representation 18, 2, 109–118.
5. MACKLIN , M., AND M ÜLLER , M. 2013. Position-based fluids. ACM Transactions on Graphics (TOG) 32, 4, 104.
6. MÜLLER , M., HEIDELBERGER , B., TESCHNER , M., AND GROSS , M. 2005. Meshless deformations based on shape matching. In ACM transactions on graphics (TOG), vol. 24, ACM, 471–478.
7. MÜLLER , M., BENDER , J., CHENTANEZ , N., AND MACKLIN, M. 2016. A robust method to extract the rotational part of deformations. In Proceedings of the 9th International Conference on Motion in Games, ACM, 55–60.
8. GREEN , S. 2010. Particle simulation using cuda.
