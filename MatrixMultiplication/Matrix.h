#include <exception>
#include <omp.h>
#include <vector>

// Note: the code below contains error checking code for use
// in debugging. To turn on error checking, uncomment
// the line below:
#define mDebug
// To turn off error checking comment the line back out.

class BadArrayAccess : public std::exception {
public:
	BadArrayAccess(int r, int c) noexcept
		: row(r), col(c) {}
	virtual const char* what() const noexcept
	{
		return "Bad array access";
	}

	int getRow() { return row; }
	int getCol() { return col; }
private:
	int row, col;
};

class IllegalViewSize : public std::exception {
public:
	IllegalViewSize(int r, int c) noexcept
		: row(r), col(c) {}
	virtual const char* what() const noexcept
	{
		return "Illegal view size";
	}

	int getRow() { return row; }
	int getCol() { return col; }
private:
	int row, col;
};

template <typename T>
class Viewable {
public:
	virtual T& operator()(int row, int col) = 0;
};

template <typename T>
class View : public Viewable<T> {
public:
	View(Viewable<T>& basedOn, int r, int c, int rows, int cols) :
		base(basedOn), rowOffset(r), colOffset(c), maxRows(rows), maxCols(cols) {}
	View(const View<T>& other)
		: base(other.base), rowOffset(other.rowOffset), colOffset(other.colOffset),
		maxRows(other.maxRows), maxCols(other.maxCols) {}
	View<T>& operator=(const View<T>& other) {
		base = other.base;
		rowOffset = other.rowOffset;
		colOffset = other.colOffset;
		maxRows = other.maxRows;
		maxCols = other.maxCols;
		return *this;
	}

	T& operator()(int row, int col) {
#ifdef mDebug
		if (row < 0 || row >= maxRows || col < 0 || col >= maxCols)
			throw BadArrayAccess(row, col);
#endif
		return base(row + rowOffset, col + colOffset);
	}

	View<T> makeView(int r, int c, int rows, int cols)
	{
#ifdef mDebug
		if (r < 0 || r + rows > maxRows || c < 0 || c + cols > maxCols)
			throw IllegalViewSize(rows, cols);
#endif
		return View<T>(*this, r, c, rows, cols);
	}
private:
	Viewable<T> &base;
	int rowOffset;
	int colOffset;
	int maxRows;
	int maxCols;
};

template <typename T>
class Matrix : public Viewable<T> {
public:
	Matrix(int r, int c) : rows(r), cols(c) {
		data = new T[r*c];
	}
	~Matrix() { delete[] data; }

	static void Multiplication(View<T>& a, View<T>& b, View<T>& c, const int size);
	static void P_Strassen(View<T>& a, View<T>& b, View<T>& c, const int size, int level);

	T& operator()(int row, int col) {
#ifdef mDebug
		if (row < 0 || row >= rows || col < 0 || col >= cols)
			throw BadArrayAccess(row, col);
#endif
		return data[row*cols + col];
	}

	View<T> makeView(int r, int c, int rowCount, int colCount)
	{
#ifdef mDebug
		if (r < 0 || r + rowCount > rows || c < 0 || c + colCount > cols)
			throw IllegalViewSize(rowCount, colCount);
#endif
		return View<T>(*this, r, c, rowCount, colCount);
	}
private:
	int rows, cols;
	T* data;

	// We declare the copy constructor and operator= to be private
	// to prevent their use in code. If you wish to use these, move
	// them to the public section of this class.
	// I strongly suggest that you avoid using these, since copies are
	// inefficient. In particular, when passing Matrix objects as 
	// parameters to functions you should always try to pass them
	// using reference parameters
	Matrix(const Matrix<T>& other) : rows(other.rows), cols(other.cols) {
		data = new T[rows*cols];
		for (int i = 0; i < rows*cols; i++)
			data[i] = other.data[i];
	}

	Matrix<T>& operator=(const Matrix<T>& other) {
		if (data)
			delete[] data;
		rows = other.rows;
		cols = other.cols;
		data = new T[rows*cols];
		for (int i = 0; i < rows*cols; i++)
			data[i] = other.data[i];
		return *this;
	}
};
template <typename T>
void Matrix<T>::Multiplication(View<T>& a, View<T>& b, View<T>& c, const int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			c(i, j) = 0;
			for (int k = 0; k < size; k++) {
				c(i, j) = c(i, j) + a(i, k) * b(k, j);
			}
		}
	}
}

template <typename T>
void Matrix<T>::P_Strassen(View<T>& a, View<T>& b, View<T>& c, const int size, int level) {
	//base case for recursion
	if (size == 1) {
		c(0, 0) = a(0, 0) * b(0, 0);
		return;
	}
	//deal with cases if row length is not 2^n
	if (size % 2 != 0) {
		Multiplication(a, b, c, size);
		return;
	}
	//limit the number of execution thread
	if (level > 1) {
		Multiplication(a, b, c, size);
		return;
	}
	int i, j;

	//generate submatrices of a,b,c
	View<T> a11 = a.makeView(0, 0, size / 2, size / 2);
	View<T> a12 = a.makeView(0, size / 2, size / 2, size / 2);
	View<T> a21 = a.makeView(size / 2, 0, size / 2, size / 2);
	View<T> a22 = a.makeView(size / 2, size / 2, size / 2, size / 2);
	View<T> b11 = b.makeView(0, 0, size / 2, size / 2);
	View<T> b12 = b.makeView(0, size / 2, size / 2, size / 2);
	View<T> b21 = b.makeView(size / 2, 0, size / 2, size / 2);
	View<T> b22 = b.makeView(size / 2, size / 2, size / 2, size / 2);
	View<T> c11 = c.makeView(0, 0, size / 2, size / 2);
	View<T> c12 = c.makeView(0, size / 2, size / 2, size / 2);
	View<T> c21 = c.makeView(size / 2, 0, size / 2, size / 2);
	View<T> c22 = c.makeView(size / 2, size / 2, size / 2, size / 2);

	//generate matrices s1~s10, p1~p7 for Strassen's algorithm
	std::vector<View<T>> s, p; 
	Matrix<T> s1(size / 2, size / 2);	s.push_back(s1.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s2(size / 2, size / 2);	s.push_back(s2.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s3(size / 2, size / 2);	s.push_back(s3.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s4(size / 2, size / 2);	s.push_back(s4.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s5(size / 2, size / 2);	s.push_back(s5.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s6(size / 2, size / 2);	s.push_back(s6.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s7(size / 2, size / 2);	s.push_back(s7.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s8(size / 2, size / 2);	s.push_back(s8.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s9(size / 2, size / 2);	s.push_back(s9.makeView(0, 0, size / 2, size / 2));
	Matrix<T> s10(size / 2, size / 2);	s.push_back(s10.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p1(size / 2, size / 2);	p.push_back(p1.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p2(size / 2, size / 2);	p.push_back(p2.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p3(size / 2, size / 2);	p.push_back(p3.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p4(size / 2, size / 2);	p.push_back(p4.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p5(size / 2, size / 2);	p.push_back(p5.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p6(size / 2, size / 2);	p.push_back(p6.makeView(0, 0, size / 2, size / 2));
	Matrix<T> p7(size / 2, size / 2);	p.push_back(p7.makeView(0, 0, size / 2, size / 2));

#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[0](i, j) = b12(i, j) - b22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[1](i, j) = a11(i, j) + a12(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[2](i, j) = a21(i, j) + a22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[3](i, j) = b21(i, j) - b11(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[4](i, j) = a11(i, j) + a22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[5](i, j) = b11(i, j) + b22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[6](i, j) = a12(i, j) - a22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[7](i, j) = b21(i, j) + b22(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[8](i, j) = a11(i, j) - a21(i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			s[9](i, j) = b11(i, j) + b12(i, j);
		}
	}
#pragma omp parallel sections
	{
#pragma omp section
		P_Strassen(a11, s[0], p[0], size / 2, level + 1);
#pragma omp section
		P_Strassen(s[1], b22, p[1], size / 2, level + 1);
#pragma omp section
		P_Strassen(s[2], b11, p[2], size / 2, level + 1);
#pragma omp section
		P_Strassen(a22, s[3], p[3], size / 2, level + 1);
#pragma omp section
		P_Strassen(s[4], s[5], p[4], size / 2, level + 1);
#pragma omp section
		P_Strassen(s[6], s[7], p[5], size / 2, level + 1);
#pragma omp section
		P_Strassen(s[8], s[9], p[6], size / 2, level + 1);
	}

#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			c11(i, j) = p[4](i, j) + p[3](i, j) - p[1](i, j) + p[5](i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			c12(i, j) = p[0](i, j) + p[1](i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			c21(i, j) = p[2](i, j) + p[3](i, j);
		}
	}
#pragma omp parallel for
	for (i = 0; i < size / 2; i++) {
#pragma omp parallel for
		for (j = 0; j < size / 2; j++) {
			c22(i, j) = p[4](i, j) + p[0](i, j) - p[2](i, j) - p[6](i, j);
		}
	}
}