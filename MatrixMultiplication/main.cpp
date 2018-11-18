#include "Matrix.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

int main() {
	int N;

	std::cout << "Give the row length N of N X N matrix: ";
	std::cin >> N;

	Matrix<int> A(N, N);
	Matrix<int> B(N, N);
	Matrix<int> C(N, N);

	//Generate elements of A and B to random integers from -100 to 100
	srand(time(0));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A(i, j) = (rand() % 201) - 100;
			B(i, j) = (rand() % 201) - 100;
		}
	}
	//print out A's element
	std::cout << "A:\n";
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << A(i, j) << " ";
		}
		std::cout << std::endl;
	}
	//print out B's element
	std::cout << "\nB:\n";
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << B(i, j) << " ";
		}
		std::cout << std::endl;
	}

	//print out C's element with P_Strassen
	std::cout << "\nC using parallel Strassen Algorithm:\n\n";
	View<int> view_A = A.makeView(0, 0, N, N);
	View<int> view_B = B.makeView(0, 0, N, N);
	View<int> view_C = C.makeView(0, 0, N, N);
	omp_set_nested(1);
	time_t now = time(NULL);
	clock_t start = clock();
	Matrix<int>::P_Strassen(view_A, view_B, view_C, N, 0);
	time_t later = time(NULL);
	clock_t end = clock();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << C(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "Parallel Strassen took " << (later-now) << " seconds.\n";
	std::cout << "CPU time was " << (end - start) / CLOCKS_PER_SEC << " seconds.\n";

	//print out C's element with simple Multiplication
	std::cout << "\nC using simple element by element multiplication:\n\n";
	now = time(NULL);
	start = clock();
	Matrix<int>::Multiplication(view_A, view_B, view_C, N);
	later = time(NULL);
	end = clock();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << C(i, j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "Simple multiplication took " << (later - now) << " seconds.\n";
	std::cout << "CPU time was " << (end - start) / CLOCKS_PER_SEC << " seconds.\n";
}