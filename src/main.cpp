#include <iostream>
#include <vector>

#include <OsqpEigen/OsqpEigen.h>
#include <matplotlibcpp.h>

const double smooth_weight = 100.0;
const double shape_weight = 1.0;
const double compact_weight = 1.0;
const double position_buffer = 0.25;

namespace plt = matplotlibcpp;

void getHessianMatrix(size_t num_pts, double w_smooth, double w_shape, double w_compact, Eigen::SparseMatrix<double>& mat_p, bool debug=false)
{
    mat_p.resize(2 * num_pts, 2 * num_pts);
    Eigen::MatrixXd mat_smooth, mat_shape, mat_compact;
    mat_smooth.resize(2 * (num_pts - 2), 2 * num_pts);
    mat_shape.resize(2 * num_pts, 2 * num_pts);
    mat_compact.resize(2 * (num_pts - 1), 2 * num_pts);

    for (int i = 0; i < num_pts - 2; ++i)
    {
        mat_smooth.block(2 * i, 2 * i, 2, 6) << 1, 0, -2, 0, 1, 0, 0, 1, 0, -2, 0, 1;
    }
    mat_shape.setIdentity();
    for (int i = 0; i < num_pts - 1; ++i)
    {
        mat_compact.block(2 * i, 2 * i, 2, 4) << 1, 0, -1, 0, 0, 1, 0, -1;
    }
    auto mat_p_dense = w_smooth * mat_smooth.transpose() * mat_smooth + w_shape * mat_shape.transpose() * mat_shape + w_compact * mat_compact.transpose() * mat_compact;
    mat_p = mat_p_dense.sparseView();
    if (debug)
    {
        std::cout << "mat smooth:\n" << mat_smooth << std::endl;
        std::cout << "mat shape:\n" << mat_shape << std::endl;
        std::cout << "mat compact:\n" << mat_compact << std::endl;
        std::cout << "mat hessian dense:\n" << mat_p_dense << std::endl;
        std::cout << "mat hessian sparse:\n" << mat_p << std::endl;
    }
}

void getGradient(double w_shape, Eigen::VectorXd& pos_ref, Eigen::VectorXd& vec_q, bool debug=false)
{
    vec_q = w_shape * (-2) * pos_ref;
    if (debug)
    {
        std::cout << "vec gradient:\n" << vec_q << std::endl;
    }
}

void getLinearConstraintsMatrix(size_t num_pts, Eigen::SparseMatrix<double>& mat_linear_constraints, bool debug=false)
{
    mat_linear_constraints.resize(num_pts * 2, num_pts * 2);
    mat_linear_constraints.setIdentity();
    if (debug)
    {
        std::cout << "mat linear constraints:\n" << mat_linear_constraints << std::endl;
    }
}

void getBounds(size_t num_pts, Eigen::VectorXd& pos_ref, double pos_buffer, Eigen::VectorXd& lower_bounds, Eigen::VectorXd& upper_bounds, bool debug=false)
{
    lower_bounds = pos_ref.array() - pos_buffer;
    upper_bounds = pos_ref.array() + pos_buffer;
    if (debug)
    {
        std::cout << "lower bounds:\n" << lower_bounds << std::endl;
        std::cout << "upper bounds:\n" << upper_bounds << std::endl;
    }
}

int main(int argc, char** argv)
{
    // test variable
    Eigen::VectorXd pos_ref(88);
    pos_ref << 0, 0, 0, 0.25, 0, 0.5, 0, 0.75,
               0, 1, 0, 1.25, 0, 1.5, 0, 1.75,
               0, 2, 0, 2.25, 0, 2.5, 0, 2.75,
               0, 3, 0, 3.25, 0, 3.5, 0, 3.75,
               0, 4, 0, 4.25, 0, 4.5, 0, 4.75,
               0, 5, 0, 5.25, 0, 5.5, 0, 5.75,
               0.125, 6, 0.25, 6.25, 0.375, 6.5, 0.5, 6.75,
               0.625, 7, 0.75, 7.25, 0.875, 7.5, 1.0, 7.75,
               1, 8, 1, 8.25, 1, 8.5, 1, 8.75,
               1, 9, 1, 9.25, 1, 9.5, 1, 9.75,
               1, 10, 1, 10.25, 1, 10.5, 1, 10.75;

    size_t num_pts = pos_ref.size() / 2;
    double w_smooth = smooth_weight, w_shape = shape_weight, w_compact = compact_weight;
    double pos_buffer = position_buffer;
    // declare QP variables
    size_t num_variables = pos_ref.size(), num_constraints = pos_ref.size();
    Eigen::SparseMatrix<double> mat_p;
    Eigen::VectorXd vec_q;
    Eigen::SparseMatrix<double> mat_linear_constraints;
    Eigen::VectorXd vec_lower_bounds, vec_upper_bounds;

    getHessianMatrix(num_pts, w_smooth, w_shape, w_compact, mat_p, false);
    getGradient(w_shape, pos_ref, vec_q, false);
    getLinearConstraintsMatrix(num_pts, mat_linear_constraints, false);
    getBounds(num_pts, pos_ref, pos_buffer, vec_lower_bounds, vec_upper_bounds, false);
    // std::cout << "hessian matrix:\n" << mat_p << std::endl;
    // std::cout << "gradient:\n" << vec_q << std::endl;
    // std::cout << "linear constraints:\n" << mat_linear_constraints << std::endl;
    // std::cout << "lower bounds:\n" << vec_lower_bounds << std::endl;
    // std::cout << "upper bounds:\n" << vec_upper_bounds << std::endl;

    // declare QP solver
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(true);
    // init QP solver
    solver.data()->setNumberOfVariables(num_variables);
    solver.data()->setNumberOfConstraints(num_constraints);
    if (!solver.data()->setHessianMatrix(mat_p)) return EXIT_FAILURE;
    if (!solver.data()->setGradient(vec_q))  return EXIT_FAILURE;
    if (!solver.data()->setLinearConstraintsMatrix(mat_linear_constraints))   return EXIT_FAILURE;
    if (!solver.data()->setLowerBound(vec_lower_bounds))    return EXIT_FAILURE;
    if (!solver.data()->setUpperBound(vec_upper_bounds))    return EXIT_FAILURE;
    if (!solver.initSolver())   return EXIT_FAILURE;
    // solve QP problem
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return EXIT_FAILURE;
    Eigen::VectorXd qp_solution = solver.getSolution();
    // std::cout << "qp solution:\n" << qp_solution << std::endl;
    // // update QP constraints
    // if (!solver.updateBounds(vec_lower_bounds, vec_upper_bounds)) return EXIT_FAILURE;
    
    // display using matplotlib
    Eigen::VectorXd x_ref(num_pts), y_ref(num_pts), x_solved(num_pts), y_solved(num_pts);
    for (int i = 0; i < num_pts; ++i)
    {
        x_ref(i) = pos_ref(i * 2);
        y_ref(i) = pos_ref(i * 2 + 1);
        x_solved(i) = qp_solution(i * 2);
        y_solved(i) = qp_solution(i * 2 + 1);
    }
    plt::figure();
    plt::plot(x_ref, y_ref, "r*-", {{"label", "ref"}});
    plt::plot(x_solved, y_solved, "b*-", {{"label", "solved"}});
    // plt::plot(x_ref, y_ref);
    plt::xlim(-1, 11);
    plt::ylim(-1, 11);
    plt::legend();
    plt::show();

    return EXIT_SUCCESS;
}