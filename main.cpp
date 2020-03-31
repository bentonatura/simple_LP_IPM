#include <iostream>
#include <Eigen/Dense>
#include <assert.h>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;


Vector solve(Matrix &M, Vector &v) {
    return M.colPivHouseholderQr().solve(v);
}

Vector all_ones_vector(const int n) {
    Vector v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = 1.0;
    }
    return v;
}

void create_initial_vector(Vector &x, double &tau, Vector &y, Vector &s, double &kappa) {
    x = all_ones_vector(x.rows());
    tau = 1;
    y = Vector::Zero(y.rows());
    s = all_ones_vector(s.rows());
    kappa = 1;
}

Matrix create_linear_system_matrix(Matrix A, Vector b, Vector c, Vector x, Vector s, double kappa, double tau) {
    const int m = A.rows();
    const int n = A.cols();

    assert(b.rows() == m);
    assert(c.rows() == n);
    assert(x.rows() == n);
    assert(s.rows() == n);

    const int dim = n + 1 + m + n + 1;
    Matrix M(dim, dim);
    M.block(0, 0, m, n) = A;
    M.block(0, n, m, 1) = -b;
    M.block(m, n, n, 1) = -c;
    M.block(m, n + 1, n, m) = A.transpose();
    M.block(m, n + 1 + m, n, n) = Matrix::Identity(n, n);
    M.block(m + n, 0,
            1, n) = -c.transpose();
    M.block(m + n, n + 1, 1, m) = b.transpose();
    M(m + n, n + 1 + m + n) = -1;
    M.block(m + n + 1, 0, n, n) = s.asDiagonal();
    M.block(m + n + 1, n + 1 + m, n, n) = x.asDiagonal();
    M(n + 1 + m + n, n) = kappa;
    M(n + 1 + m + n, n + 1 + m + n) = tau;
    return M;
}

int main(int const argc, char **argv) {
    srand((unsigned int) time(0));
    int m = 5;
    int n = 10;
    if(argc > 2){
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
    }
    const int max_num_iterations = 50;
    const double max_step_length = 0.5;

    //Assume A has full row rank
    Matrix A = Matrix::Random(m, n);
    Vector x0 = Vector::Random(n);
    Vector c = Vector::Random(n);
    //for loop to create a feasible and bounded instance
    for (int i = 0; i < n; i++) {
        x0(i) += 1;
        c(i) += 1;
    }
    Vector b = A * x0;
    Vector x(n), s(n), y(m);
    double tau, kappa;
    create_initial_vector(x, tau, y, s, kappa);

    //Error vectors
    Vector r_p(m), r_d(n), r_xs(n);
    double r_g, mu, r_tau_kappa;

    for (int iter = 0; iter < max_num_iterations; iter++) {

        Vector d(n + 1 + m + n + 1);
        Vector d_x(n), d_tau(1), d_y(m), d_s(n), d_kappa(1);

        r_p = tau * b - A * x;
        r_d = tau * c - A.transpose() * y - s;
        r_g = kappa + c.dot(x) - b.dot(y);
        mu = (x.dot(s) + tau * kappa) / (n + 1);

        r_xs = mu * all_ones_vector(n) - x.asDiagonal() * (s);
        r_tau_kappa = mu - tau * kappa;

        Vector r(r_p.rows() + r_d.rows() + 1 + x.rows() + 1);
        r << r_p, r_d, r_g, r_xs, r_tau_kappa;

        Matrix M = create_linear_system_matrix(A, b, c, x, s, kappa, tau);
        d = solve(M, r);

        d_x = d.segment(0, n);
        d_tau = d.segment(n, 1);
        d_y = d.segment(n + 1, m);
        d_s = d.segment(n + 1 + m, n);
        d_kappa = d.segment(n + 1 + m + n, 1);

        //find max step length to maintain non-negativity

        double x_min = (x.array().inverse() * d_x.array()).minCoeff();
        double s_min = (s.array().inverse() * d_s.array()).minCoeff();
        double kappa_min = d_kappa(0) / kappa;
        double tau_min = d_tau(0) / tau;
        double min_ratio = -std::min({x_min, s_min, kappa_min, tau_min});
        double alpha_max = 1 / min_ratio;

        //steplength
        double const alpha = std::min(max_step_length, alpha_max);

        x += alpha * d_x;
        s += alpha * d_s;
        tau += alpha * d_tau(0);
        y += alpha * d_y;
        kappa += alpha * d_kappa(0);

//        Matrix xs_matrix(n, 2);
//        xs_matrix.block(0, 0, n, 1) = x/tau;
//        xs_matrix.block(0, 1, n, 1) = s/tau;
//        std::cout << "x and s after " << iter << " iterations." << std::endl;
//        std::cout << " x s: " << std::endl << xs_matrix << std::endl;
    }

    //Set optimal solution (We cheat, because we know that the primal is feasible and bounded and therefore both primal
    //and dual are feasible and an optimal partition exists)

    x /= tau;
    s /= tau;
    y /= tau;

    //Check correctness:

    std::cout << "Near feasible and optimal solutions:" << std::endl;

    Matrix xs_matrix_final(n, 2);
    xs_matrix_final.block(0, 0, n, 1) = x;
    xs_matrix_final.block(0, 1, n, 1) = s;
    std::cout << " x s: " << std::endl << xs_matrix_final << std::endl;

//    std::cout << "tau: " << tau << " kappa: " << kappa << std::endl;
    std::cout << std::endl << "Check validity..." << std::endl;
    std::cout << "Primal error: " << (A * x - b).norm() << std::endl;
    std::cout << "Dual error: " << (A.transpose() * y + s - c).norm() << std::endl;
    std::cout << "complementarity error : " << (x.asDiagonal() * s).norm() << std::endl;
    std::cout << "Primal non-negativity: " << (x.array().min(Vector::Zero(n).array())).matrix().norm() << std::endl;
    std::cout << "Dual non-negativity: " << (s.array().min(Vector::Zero(n).array())).matrix().norm() << std::endl;


    return 0;
}
