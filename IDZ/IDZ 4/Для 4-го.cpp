#include <iostream>
#include <vector>

using namespace std;

template <typename T>
class Matrix {
private:
    vector<vector<T>> data;
    size_t N; // number of rows
    size_t M; // number of columns
public:
    Matrix(size_t row, size_t column) : M(column), N(row),
                                        data(row, vector<T>(column)) {}

    Matrix(const vector<vector<T>>& my_vector) : data(my_vector),
                                                 M(my_vector[0].size()),
                                                 N(my_vector.size()) {}

    Matrix& operator+=(const Matrix& other) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j){
                data[i][j] += other.data[i][j];
            }
        }
        return *this;
    }

    Matrix operator+(const Matrix& other) const {
        Matrix temp(*this);
        temp += other;
        return temp;
    }

    Matrix& operator-=(const Matrix& other) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j){
                data[i][j] -= other.data[i][j];
            }
        }
        return *this;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix temp(*this);
        temp -= other;
        return temp;
    }


    Matrix operator*=(const Matrix& other) {
        Matrix temp(N, other.M);
        if (M != other.N) {
            cout << "Error! This Matrixes can't be multiplied \n";
            return *this;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < other.M; ++j) {
                T value = 0;
                for (size_t l = 0; l < M; ++l){
                    value += data[i][l] * other.data[l][j];
                }
                temp.data[i][j] = value;
            }
        }

        *this = temp;
        return *this;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix temp(*this);
        temp *= other;
        return temp;
    }

    vector<T>& operator[](size_t i) {
        return data[i];
    }

    size_t row_count() const {
        return N;
    }

    size_t column_count() const {
        return M;
    }

    void swap_matrix(string param, size_t z1, size_t z2) { // Swap two row or columns
        if (param == "st") {                              // Param == str is for
            for (size_t j = 0; j < M; ++j)                 // elementary operation for row
                swap(data[z1][j],  data[z2][j]);
        } else {
            for (size_t i = 0; i < N; ++i)
                swap(data[i][z1],  data[i][z2]);
        }
    }

    void add(string param, size_t to, size_t val, T coefficient) { // Add one row(column), miltiplied to
        if (param == "st") {                                      // coefficient to other row(column)
            for (size_t j = 0; j < M; ++j)
                data[to][j] += data[val][j] * coefficient;
        } else {
            for (size_t i = 0; i < N; ++i)
                data[i][to] += data[i][val] * coefficient;
        }
    }

    void mult(string param, size_t z, T coefficient) { // Multiplying row(column) to coefficient
        if (param == "st") {
            for (size_t j = 0; j < M; ++j)
                data[z][j] *= coefficient;
        } else {
            for (size_t i = 0; i < N; ++i)
                data[i][z] *= coefficient;
        }
    }

    void div(string param, size_t z, T coefficient) { // Dividing row(column) to coefficient
        if (param == "st") {
            for (size_t j = 0; j < M; ++j) {
                if (data[z][j] % coefficient > 0) {
                    cout << "Attention!" << endl;  // Warning, when the number isn't divisible
                }
                data[z][j] /= coefficient;
            }
        } else {
            for (size_t i = 0; i < N; ++i) {
                if (data[i][z] % coefficient > 0) {
                    cout << "Attention!" << endl;
                }
                data[i][z] /= coefficient;
            }
        }
    }

    void transposition() {
        Matrix temp(M, N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < M; ++j) {
                temp[j][i] = data[i][j];
            }
        }
        *this = temp;
    }

    void cout_in_latex() const {  // Couting in format, fitted for just cope-paste to latex code
        cout << "\\[" << endl << "\\begin{pmatrix}" << endl;
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < M; ++j) {
                cout << data[i][j];
                if (j == M - 1)
                    cout << " \\\\";
                else
                    cout << " & ";

                /*if (j == 2) {
                    cout << "| & ";  //To make block matrix
                }*/
            }
            cout << endl;
        }
        cout << "\\end{pmatrix}" << endl << "\\]" << endl ;

        cout << endl;
    }

};

template <typename T>
void diag_assign(Matrix<T>& a, const T& value) { // Make a matrix diagonal view with value on main diagonal
    if (a.row_count() != a.column_count()) {
        cout << "Error! This matrix isn't sqare" << endl;
        return;
    }
    for (size_t i = 0; i < a.row_count(); ++i) {
        a[i][i] = value;
    }
}

template <typename T>
ostream& operator<<( ostream& os, Matrix<T>& a) {
    for (size_t i = 0; i < a.row_count(); ++i) {
        for (size_t j = 0; j < a.column_count(); ++j) {
            cout << a[i][j] << '\t';
        }
        cout << endl;
    }
    return os;
}

int main()
{
    freopen("in","r",stdin);
    freopen("out","w",stdout);

    int n, m;
    cin >> n >> m;
    vector<vector<double>> v(n, vector<double>(m));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            cin >> v[i][j];
        }

    Matrix<double> A(v);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            cin >> v[i][j];
        }

    Matrix<double> S1(v);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            cin >> v[i][j];
        }

    Matrix<double> S2(v);

    Matrix<double> S = S1 * S2;


    string s;

    S.transposition();
    Matrix<double> ST(S);
    S.transposition();

    cout << ST << endl << A << endl << S << endl;

    Matrix<double> ANS = ST * A * S;

    cout << ANS;

    return 0;

}
