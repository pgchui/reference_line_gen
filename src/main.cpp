#include <OsqpEigen/OsqpEigen.h>
#include <iostream>

const double smooth_weight = 1.0;
const double shape_weight = 0.1;
const double compact_weight = 10.0;
const double position_buffer = 0.5;

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
    Eigen::VectorXd pos_ref(8);
    pos_ref << 0, 0, 0, 1, 0, 2, 0, 4;

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
    std::cout << "qp solution:\n" << qp_solution << std::endl;
    // // update QP constraints
    // if (!solver.updateBounds(vec_lower_bounds, vec_upper_bounds)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}