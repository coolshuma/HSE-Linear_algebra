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

//                if (j == 3) {
//                    cout << "| & ";  //To make block matrix
//                }
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
    vector<vector<int>> v(n, vector<int>(m));

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            cin >> v[i][j];
        }

    Matrix<int> a(v);

    a.cout_in_latex();

    string s;

    //cout << a << endl;

    while (cin >> s) {
        string type;
        cin >> type;

        if (s == "swap")
        {
            size_t  z1, z2;
            cin >> z1 >> z2;
            z1--; z2--;
            a.swap_matrix(type, z1, z2);
            cout << "Поменяем местами " << z1 + 1;
            if (type == "st")
                cout << "-ую и "  << z2 + 1 << "-ую строку";
            else
                cout << "-ый и " << z2 + 1 << "-ый столбец";

            cout << ':' << endl;

        }

        if (s == "add")
        {
            int z1, z2; int coefficient;
            cin >> z1 >> z2 >> coefficient;
            z1--; z2--;
            a. add(type, z1, z2, coefficient);
            cout << "Добавим к " << z1 + 1;
            if (type == "st")
                cout << "-ой строке "  << z2 + 1 << "-ую, домноженную на ";
            else
                cout << "-ому столбцу " << z2 + 1 << "-ый, домноженный на ";

            cout << coefficient << ':' << endl;
        }

        if (s == "div")
        {
            int z1; int coefficient;
            cin >> z1 >> coefficient;
            z1--;
            a.div(type, z1, coefficient);
            cout << "Разделим " << z1 + 1;
            if (type == "st")
                cout << "-ую строку на ";
            else
                cout << "-ый столбец на ";

            cout << coefficient << ':' << endl;
        }

        if (s == "mult")
        {
            int z1; int coefficient;
            cin >> z1 >> coefficient;
            z1--;
            a.mult(type, z1, coefficient);
            cout << "Домножим " << z1 + 1;
            if (type == "st")
                cout << "-ую строку на ";
            else
                cout << "-ый столбец на ";

            cout << coefficient << ':' << endl;
        }

        cout << endl;
        //cout << a << endl;

        a.cout_in_latex();
    }

    //a.transposition();

    //a.cout_in_latex();

    return 0;
}